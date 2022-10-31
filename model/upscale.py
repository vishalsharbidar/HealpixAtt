import torch
import torch.nn as nn
import time
from pooling.calculating_cog import Calculating_COG, Center_of_Gravity
from sklearn.neighbors import KDTree
import numpy
from model.superglue import arange_like, log_optimal_transport
from model.loss import criterion

def pairwise_cosine_dist(x1, x2):
    """
    Return pairwise half of cosine distance in range [0, 1].
    dist = (1 - cos(theta)) / 2
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return 0.25 * torch.cdist(x1, x2).pow(2)


class Upscale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bin_score = torch.nn.Parameter(torch.tensor(0.)) 

    def finding_valid_child_facets_idx_for_source_img(self, NSIDE, corr_pred_mask, output):    
        # Extracting the index of facets with correspondence
        kptidx_with_corr_img0 = torch.masked_select(torch.arange(corr_pred_mask.shape[1]).to(self.config['device']), corr_pred_mask).to(torch.int)
        # Extracting the parent facets with correspondence
        img0_facetsidx = torch.index_select(output[NSIDE]['img0_facetsidx'], 0, kptidx_with_corr_img0)
        # Retriving child facets for given parent 
        img0_child_facets = torch.cat([4*img0_facetsidx.view(-1, 1) + i for i in range(4)], dim=1).view(-1,1)
        # Loading the child facet got from pooling
        img0_child_facets_from_hh_pool = output[NSIDE]['img0_child_facets']
        # Creating a mask to find a valid facets in img0_child_facets
        img0_mask = torch.isin(img0_child_facets, img0_child_facets_from_hh_pool)
        # Extracting the valid facets
        valid_img0_child_facets = torch.mul(img0_child_facets, img0_mask).int().view(-1)
        # Load mapping dict to trace index of child facets 
        img0_map_dict_child_facetlabel_per_keypt = output[NSIDE]['img0_map_dict_child_facetlabel_per_keypt']
        # Extracting the valid indices for child facets
        valid_img0_facets_idx = torch.tensor(list(map(img0_map_dict_child_facetlabel_per_keypt.get, list(valid_img0_child_facets.cpu().numpy())))) 
        return valid_img0_facets_idx 

    def finding_valid_child_facets_idx_for_target_img(self, NSIDE, correspondence, corr_pred_mask, output, k):    
        # Extracting the index of facets with correspondence
        kptidx_with_corr_img1 = torch.masked_select(correspondence, corr_pred_mask).to(torch.int64)   
        # Extracting the parent position with correspondence    
        img1_parent_position_with_corr = torch.index_select(output[NSIDE]['img1_parent_position'], 1, kptidx_with_corr_img1)
        # Loading the healpix KD tree
        healpix_kdtree =  self.config['kdtree'][NSIDE//2]
        # Extracting the parent k nearest neighbours to find correspondence    
        #print(img1_parent_position_with_corr.shape, img1_parent_position_with_corr)
        #exit()
        img1_facets_with_20_nn = torch.tensor(healpix_kdtree.query(img1_parent_position_with_corr[0].cpu(), k=k, return_distance=False)).squeeze().to(self.config['device'])               
        # Retriving child facets for given parents 
        img1_child_facets = torch.cat([4*img1_facets_with_20_nn.view(-1, 1) + i for i in range(4)], dim=1).view(-1,1)
        # Loading the child facet got from pooling
        img1_child_facets_from_hh_pool = output[NSIDE]['img1_child_facets']
        # Creating a mask to find a valid facets in img0_child_facets
        img1_mask = torch.isin(img1_child_facets, img1_child_facets_from_hh_pool)
        #Extracting the valid facets
        valid_img1_child_facets = torch.mul(img1_child_facets, img1_mask).int().view(-1)
        # Load mapping dict to trace index of child facets 
        img1_map_dict_child_facetlabel_per_keypt = output[NSIDE]['img1_map_dict_child_facetlabel_per_keypt']
        # Extracting the valid indices for child facets
        valid_img1_facets_idx = torch.tensor(list(map(img1_map_dict_child_facetlabel_per_keypt.get, list(valid_img1_child_facets.cpu().numpy()))))
        return valid_img1_facets_idx
    
    def creating_upscaled_scores_and_mask_matrix(self, NSIDE, output):
        upscaled_scores = torch.einsum('bdn,bdm->bnm', output[NSIDE]['img0_child_descriptor'].transpose(2, 1), output[NSIDE]['img1_child_descriptor'].transpose(2, 1)) 
        return upscaled_scores
    
    def creating_mask_matrix_for_upscaled_scores_matrix(self, upscaled_scores, valid_img0_child_facets_idx, valid_img1_child_facets_idx):
        # Extracting the index of facets where keypoints is present
        img0_facet_idx = torch.where(valid_img0_child_facets_idx>0)
        img1_facet_idx = torch.where(valid_img1_child_facets_idx>0)
        # Creating a batch to see which facets from source matches the facets in target image 
        img0_batch_idx = torch.div(img0_facet_idx[0], 4, rounding_mode='trunc')
        img1_batch_idx = torch.div(img1_facet_idx[0], 80, rounding_mode='trunc')

        # Finding the probable matches
        corresponding_idx = torch.tensor([])
        for ix, ele in enumerate(img0_batch_idx):
            target_img_idx = valid_img1_child_facets_idx[img1_facet_idx][torch.where(img1_batch_idx==ele)]
            source_ing_idx = valid_img0_child_facets_idx[img0_facet_idx][ix].item() * torch.ones(target_img_idx.shape[0])
            stacked_idx = torch.stack((source_ing_idx,target_img_idx),1)
            corresponding_idx = torch.cat((corresponding_idx,stacked_idx),0).int()

        img0_idx, img1_idx = torch.split(corresponding_idx,1,dim=1) 
    
        img0_idx = list(img0_idx.view(-1).numpy())
        img1_idx = list(img1_idx.view(-1).numpy())
        batch_idx = list(torch.zeros(corresponding_idx.shape[0]).int().numpy())
        
        # Creating a Mask matrix
        a,b,c = upscaled_scores.shape
        mask_matrix = torch.zeros(a,b,c)
        # Inserting 1 where child facets have matches 
        mask_matrix[batch_idx, img0_idx, img1_idx] = 1
        
        return mask_matrix.to(self.config['device'])

    def finding_upscaled_scores_and_mask_matrix_for_given_nside(self, NSIDE, output, indices0):
        # Creating score matrix using the child descriptor for given nside and a mask matrix 
        upscaled_scores = self.creating_upscaled_scores_and_mask_matrix(NSIDE, output)
        # Extracting the correspondence 
        correspondence = indices0
        predicted_correspondence_mask = correspondence.ge(0) # Should use the correspondence predicted from superglue for nside 32
        # Finding valid facet index for source and target images
        valid_img0_child_facets_idx = self.finding_valid_child_facets_idx_for_source_img(NSIDE, predicted_correspondence_mask, output)     
        valid_img1_child_facets_idx = self.finding_valid_child_facets_idx_for_target_img(NSIDE, correspondence, predicted_correspondence_mask, output, k=20)
        
        # Creating mask matrix
        mask_matrix = self.creating_mask_matrix_for_upscaled_scores_matrix(upscaled_scores, valid_img0_child_facets_idx, valid_img1_child_facets_idx)
        # Element wise multiplication of upscaled scores matrix with mask matrix
        upscaled_scores = torch.mul(upscaled_scores, mask_matrix)

        ################ self and cross attn #####################
        desc_dim = output[NSIDE]['img0_child_descriptor'].shape[2]
        upscaled_scores = upscaled_scores / desc_dim**.5      
        
        # Run the optimal transport.
        upscaled_scores_ot = log_optimal_transport(upscaled_scores, self.bin_score, iters=20)
        #print('upscaled_scores ot',upscaled_scores_ot.shape)
        #print(upscaled_scores)
        ########################

        # Find index 
        max0, max1 = upscaled_scores_ot[:, :-1, :-1].max(2), upscaled_scores_ot[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        zero = upscaled_scores_ot.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        
        return upscaled_scores_ot, upscaled_scores, indices0

    def cr_loss(self, NSIDE, upscale, gt_corr):
        y_pred = {'context_descriptors0': None,
                'upscaled_scores': upscale['upscaled_scores'][NSIDE],
                'scores': upscale['upscaled_scores_ot'][NSIDE],
                'matches0': upscale['indices0'][NSIDE],  # use -1 for invalid match
                }
                
        y_true = {}
        y_true['gt_matches0'] = gt_corr['gt_matches0'][NSIDE].long()
        y_true['gt_matches1'] = gt_corr['gt_matches1'][NSIDE].long().unsqueeze(0)
        loss = criterion(y_true, y_pred, margin=0)
        return loss

    def forward(self, indices0, gt_corr, output, scores): # replace gt_corr with indices0
        upscale = {}
        upscale['upscaled_scores'] = {}
        upscale['upscaled_scores_ot'] = {}
        upscale['mask_matrix'] = {}
        upscale['indices0'] = {16:indices0}
        loss = {}

        if indices0.unique().shape[0]>1:
            t1 = time.time()
            NSIDE = self.config['nsides'][3]
            upscale['upscaled_scores_ot'][NSIDE], upscale['upscaled_scores'][NSIDE], upscale['indices0'][NSIDE] = self.finding_upscaled_scores_and_mask_matrix_for_given_nside(NSIDE, output, indices0)
            loss[NSIDE] = self.cr_loss(NSIDE, upscale, gt_corr)
            return upscale, loss
        else:
            return None, None

            #if upscale['indices0'][NSIDE].unique().shape[0]>2:
             #   print(ok)
              #  NSIDE = NSIDE = self.config['nsides'][2]
               # upscale['upscaled_scores_ot'][NSIDE], upscale['upscaled_scores'][NSIDE], upscale['indices0'][NSIDE] = self.finding_upscaled_scores_and_mask_matrix_for_given_nside(NSIDE, output, upscale['indices0'][NSIDE//2])
                #loss[NSIDE] = self.loss(upscale['upscaled_scores'][NSIDE][0], gt_corr['gt_matches0'][NSIDE][0].long(), gt_corr['gt_matches1'][NSIDE].long())    
                

                #NSIDE = NSIDE = self.config['nsides'][1]
                #upscale['upscaled_scores_ot'][NSIDE], upscale['upscaled_scores'][NSIDE], upscale['indices0'][NSIDE] = self.finding_upscaled_scores_and_mask_matrix_for_given_nside(NSIDE, output, upscale['indices0'][NSIDE//2])
                #loss[NSIDE] = self.loss(upscale['upscaled_scores'][NSIDE][0], gt_corr['gt_matches0'][NSIDE][0].long(), gt_corr['gt_matches1'][NSIDE].long())

                #NSIDE = NSIDE = self.config['nsides'][0]
                #upscale['upscaled_scores_ot'][NSIDE], upscale['upscaled_scores'][NSIDE], upscale['indices0'][NSIDE] = self.finding_upscaled_scores_and_mask_matrix_for_given_nside(NSIDE, output, upscale['indices0'][NSIDE//2])
                #loss[NSIDE] = self.loss(upscale['upscaled_scores'][NSIDE][0], gt_corr['gt_matches0'][NSIDE][0].long(), gt_corr['gt_matches1'][NSIDE].long())
                #t2 = time.time()
                
                #print('time', t2-t1)
                #print(upscale['indices0'][16].unique().shape, upscale['indices0'][16].unique())
                #print(upscale['indices0'][32].unique().shape, upscale['indices0'][32].unique())
                #print(upscale['indices0'][64].unique().shape, upscale['indices0'][64].unique())
                #print(upscale['indices0'][128].unique().shape, upscale['indices0'][128].unique())
                #print(upscale['indices0'][256].unique().shape, upscale['indices0'][256].unique())
                #exit()
            
        
        
        

#image0_path = 'image/00000004.jpg'
#image1_path = 'image/00000191.jpg'
#keypointCoords0 = img0 #data['keypointCoords0'].squeeze().detach().cpu().numpy()
#keypointCoords1 = img1
#matches0 = out_128['facets_corr'][0].detach().cpu()
#plot(matches_between_two_images(image0_path, image1_path, keypointCoords0, keypointCoords1, matches0))
#exit()

       