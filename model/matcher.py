# %BANNER_BEGIN%
# ------------------------------------------------------------------------------------
# 
# 
# 
# 
# 
#
# ------------------------------------------------------------------------------------
# %AUTHORS_BEGIN%
# 
# Creator: Vishal Sharbidar Mukunda
#
# %AUTHORS_END%
# ------------------------------------------------------------------------------------
# %BANNER_END%

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from model.SphericalNet import HealpixHierarchy
from model.superglue import AttentionalGNN, KeypointEncoder, log_optimal_transport, arange_like
from pooling.finding_correspondence import Finding_Correspondence
from model.upscale import Upscale
from model.loss import criterion
    
#####...........................Feature Extractor.....................................#####

class FeatureExtractor(nn.Module):

    def __init__(self, in_dim, ks, dim, drop_p, K, aggr):
        super(FeatureExtractor, self).__init__()
        self.first_gcn = ChebConv(in_channels=in_dim, out_channels=dim, K=K, normalization='sym', aggr=aggr)
        self.drop_p = drop_p

    def forward(self, h, edges):
        h = F.dropout(h, p=self.drop_p, training=self.training)
        h = self.first_gcn(h, edges)
        h = F.elu(h)
        h, idx = self.g_unet(h, edges)
        return h, idx

#####........................... Cross Attention ..........................................#####

class Attention(nn.Module):

    def __init__(self, dim, GNN_layers, config):
        super(Attention, self).__init__()
        self.config = config
        self.attn = AttentionalGNN(dim, GNN_layers*1) 

    def forward(self, desc1, desc2):
        attndesc1, attndesc2 = self.attn(desc1, desc2)
        return attndesc1, attndesc2

#####...........................Spherical Matching .....................................#####

# Matching of keypoints
class SphericalMatching(nn.Module):
    
    def __init__(self, config):
        super(SphericalMatching, self).__init__()
        self.config = config
        mid_encoder = [254]
        self.keypoint_encoder = KeypointEncoder(self.config['descriptor_dim'], mid_encoder )
        self.healpix_hierarchy = HealpixHierarchy(in_channels=self.config['descriptor_dim'], out_channels=self.config['output_dim'], K=2, aggr='mean', config=self.config)
        self.attention = Attention(self.config['descriptor_dim'], self.config['GNN_layers'], self.config)
        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1, bias=True) 
        self.bin_score = torch.nn.Parameter(torch.tensor(0.)) 
        
        self.finding_correspondence = Finding_Correspondence(self.config)
        for param in self.finding_correspondence.parameters():
            param.require_grad=False
            
        self.upscale = Upscale(self.config)

        if self.config['mode'] == 'test':
            path = 'HealpixGlue-transfer_learning/saved_model/best_model_parameters.pth'
            self.load_state_dict(torch.load(path))
            print('Loaded SphericalMatching model weights')
        
    # change from 1->0, 2->1
    def forward(self, data):

        if self.config['kpt_encoder'] == 'True': 
            # Keypoint MLP encoder.
            desc0 = self.keypoint_encoder(data['keypointDescriptors0'], data['keypointCoords0'], data['keypointScores0']) 
            desc1 = self.keypoint_encoder(data['keypointDescriptors1'], data['keypointCoords1'], data['keypointScores1']) 
            #print(desc0.transpose(1,2).shape, data['keypointDescriptors0'].shape)
            #exit()

            # SphericalGraphUNet 
            output, hh_desc0, hh_desc1 = self.healpix_hierarchy(desc0, desc1, data) 

        else:
            # SphericalGraphUNet 
            output, hh_desc0, hh_desc1 = self.healpix_hierarchy(data['keypointDescriptors0'], data['keypointDescriptors1'], data) 

        # Calculating score matrix (Compute matching descriptor distance) for given pair of images.
        if self.config['attn_layer'] == 'True': 
            # Multi-layer Transformer network.
            attndesc0, attndesc1 = self.attention(hh_desc0[32].transpose(1,2), hh_desc1[32].transpose(1,2))
            
            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(attndesc0), self.final_proj(attndesc1)
           
        else:
            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(hh_desc0[32]), self.final_proj(hh_desc1[32])

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)  # MxN
        scores = scores / self.config['output_dim']**.5      
        
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score, 
            iters=self.config['sinkhorn']) #remains same
        #print(scores.shape)
    
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        
        # Calculating groundtruth correspondence for different nsides
        with torch.no_grad():
            gt_corr = self.finding_correspondence(output, data)

        y_pred = {
            'context_descriptors0': mdesc0,
            'context_descriptors1': mdesc1,
            'scores': scores,
            'matches0': indices0,  # use -1 for invalid match
            'matching_scores0': mscores0,
             }
             
        y_true = {}
        y_true['gt_matches0'] = gt_corr['gt_matches0'][16]
        y_true['gt_matches1'] = gt_corr['gt_matches1'][16].unsqueeze(0)
        y_pred['total_loss'] = criterion(y_true, y_pred, margin=0)
        
        healpix_output = {'output':output, 
                        'hh_desc0':hh_desc0, 
                        'hh_desc1':hh_desc1}

        return y_pred, gt_corr, output


    
    # ---------------------------------- Upscale ----------------------------------------------*/

class SphericalMatching_32(nn.Module):
    
    def __init__(self, config):
        super(SphericalMatching_32, self).__init__()
        self.config = config
        self.course_matcher = SphericalMatching(self.config)
        self.upscale = Upscale(self.config)

        #path = 'HealpixGlue/model/weights/best_model_parameters.pth'
        #self.course_matcher.load_state_dict(torch.load(path))
        #print('Loaded SphericalMatching model weights')

        #for param in self.course_matcher.parameters():
        #    param.require_grad=False

    def forward(self, data):
        #with torch.no_grad():
        y_pred, gt_corr, healpix_output = self.course_matcher(data) 
            
        scaling, scaled_loss = self.upscale(y_pred['matches0'], gt_corr, healpix_output, y_pred['scores']) # replace gt_corr with indices0
        print(scaling, scaled_loss[32], y_pred, gt_corr)
        exit()
        return scaling, scaled_loss[32], y_pred, gt_corr


    # ----------------------------------END----------------------------------------------*/



'''
t4 = time.time()
#print('till scaling time',t4-t1)
#scaling, scaled_loss = self.upscale(indices0, gt_corr, output, hh_desc0, hh_desc1, scores) # replace gt_corr with indices0
t5 = time.time()
#print('scaling time',t5-t4)


scaled_loss = None
if scaled_loss != None:
    y_pred['total_loss'] = 0.9*loss + 0.1*scaled_loss[32] 
    #y_pred['total_loss'] = 0.9*loss + 0.025*scaled_loss[32] + 0.025*scaled_loss[64] + 0.025*scaled_loss[128] + 0.025*scaled_loss[256]
    #print('\nloss',loss.item(), scaled_loss[32].item() + scaled_loss[64].item() + scaled_loss[128].item() + scaled_loss[256].item())
else:
    y_pred['total_loss'] = loss
    
'''