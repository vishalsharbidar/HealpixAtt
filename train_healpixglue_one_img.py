import torch
import torchvision
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy
from numpy import asarray
from model.matcher import SphericalMatching, SphericalMatching_32
from model.loss import criterion
import json
from utils.Utils import sphericalToCartesian, find_unique_ele
from utils.ploting import matches_between_two_images, plot, keypoints_on_healpix_map_over_Single_image
from torch_geometric.nn import knn_graph
torch.cuda.empty_cache()
import os
import time
from utils.FacetsMapping import HealpyFacetsMappingDict
import healpy as hp
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tb_path/single_pair')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SphericalMatching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, nargs='+', default='data/',
        help=' Input path', metavar='')
    parser.add_argument('--nsides', nargs='+', type=float, default= [256, 128, 64, 32, 16, 8],
        help=' Healpix nside ', metavar='')
    parser.add_argument('--output_dim', type=int, default=1024,
        help=' Output dimension ', metavar='')
    parser.add_argument('--drop_p', type=float, default=0,
        help=' Dropout value ', metavar='')
    parser.add_argument('--match_threshold', type=float, default=0.5,
        help=' Match threshold ', metavar='')
    parser.add_argument('--knn', type=int, default=50,
        help=' K nearest neighbour for creating edges', metavar='')
    parser.add_argument('--K', type=int, default=2,
        help=' Chebyshev filter size K', metavar='')
    parser.add_argument('--epoch', type=int, default=450,
        help=' Number of epochs', metavar='')
    parser.add_argument('--descriptor_dim', type=int, default=128,
        help=' Dimension of descriptor', metavar='') # 64, 128, 256
    parser.add_argument('--sinkhorn', type=int, default=5,
        help=' Sinkhorn iterations', metavar='')
    parser.add_argument('--GNN_layers', type=str, nargs='+', default=['cross'],
        help=' GNN layers', metavar='') # ['self', 'cross']
    parser.add_argument('--mode', type=str, nargs='+', default='train', # ['train', 'eval']
        help=' Model mode', metavar='') 
    parser.add_argument('--kpt_encoder', type=str, nargs='+', default='True',
        help=' Keypoint encoder', metavar='')
    parser.add_argument('--attn_layer', type=str, nargs='+', default='True',
        help=' Attention Layer', metavar='')
    parser.add_argument('--aggregation', type=str, nargs='+', default='max',
        help=' Aggregation', metavar='') #['add', 'mean', 'max']
    parser.add_argument('--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    args = parser.parse_args()
    print('\nargs',args)
    # Connecting device to cuda or cpu
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print('\nRunning inference on device \"{}\"'.format(device))
    
    # Healpy Facets Mapping 
    facets_mapping = HealpyFacetsMappingDict().to(device)
    healpy_config = facets_mapping(args.nsides, device)
    
    # Config file
    default_config = {'input_dim': args.descriptor_dim,
                    'nsides': args.nsides, 
                    'output_dim':args.output_dim, 
                    'drop_p': args.drop_p, 
                    'K': args.K, 
                    'match_threshold': args.match_threshold,
                    'descriptor_dim':args.descriptor_dim, 
                    'GNN_layers':args.GNN_layers, 
                    'sinkhorn':args.sinkhorn,
                    'kpt_encoder': args.kpt_encoder,
                    'attn_layer': args.attn_layer,
                    'aggr': args.aggregation,
                    'knn':args.knn,
                    'device': device,
                    'mode': args.mode,
                }

    default_config.update(healpy_config)

    ###########################################  

    def get_correspondence(stacked_corr, facets_idx_with_kpts_img0, facets_idx_with_kpts_img1, config):
        a,b = torch.split(stacked_corr, 1,1)
        _, idx0 = torch.where(a==facets_idx_with_kpts_img0.to(config['device']))
        _, idx1 = torch.where(b==facets_idx_with_kpts_img1.to(config['device']))
        corr = -1*torch.ones(facets_idx_with_kpts_img0.shape[0]).to(config['device'], dtype=idx1.dtype)
        corr.scatter_(0, idx0, idx1)
        return corr  

    def UnitCartesian(points, device):  
        phi, theta =  torch.split(torch.as_tensor(points, dtype=torch.float), 1, dim=1)   
        unitCartesian = sphericalToCartesian(phi, theta, 1)        
        return unitCartesian.squeeze(-1).to(device)

    def img2_corr(keypointCoords1, correspondences, device):
        target = -1*torch.ones(len(keypointCoords1))
        for ii, ix_2 in enumerate(correspondences):    
            if ix_2.item() != -1:
                target[int(ix_2.item())]=int(ii)
        return target.to(device)
    
    def kpt_equalizer(correspondences):
        for ii, ix_2 in enumerate(correspondences):
            if ix_2 > 11792:
                correspondences[ii]= -1
        return correspondences
        
    def __facet_label(NSIDE,keypointCoords0, keypointCoords1, img0_corr, img1_corr, config):        
        
        device = config['device']       
        # Finding in which facet, keypoint is located
        x0,y0,z0 = numpy.split(keypointCoords0.cpu().numpy(),3,1)
        x1,y1,z1 = numpy.split(keypointCoords1.cpu().numpy(),3,1)
        facets_idx_with_kpts_img0 = torch.from_numpy(hp.vec2pix(NSIDE, x0,y0,z0, nest=True)).view(-1)
        facets_idx_with_kpts_img1 = torch.from_numpy(hp.vec2pix(NSIDE, x1,y1,z1, nest=True)).view(-1)
        mask = img0_corr.ge(0)
        kptidx_with_corr_img0 = torch.masked_select(torch.arange(mask.shape[0]).to(config['device']), mask).to(torch.int)
        kptidx_with_corr_img1 = torch.masked_select(img0_corr, mask).to(torch.int64)  
        facets_idx_with_kpts_img0_with_corr = torch.index_select(facets_idx_with_kpts_img0.to(config['device']), 0, kptidx_with_corr_img0)
        facets_idx_with_kpts_img1_with_corr = torch.index_select(facets_idx_with_kpts_img1.to(config['device']), 0, kptidx_with_corr_img1)
        stacked_corr = torch.stack((facets_idx_with_kpts_img0_with_corr, facets_idx_with_kpts_img1_with_corr), 1)
        
        # Fining unique facets and its original index from dataset
        facets_idx_with_kpts_img0, original_idx_img0, seen = find_unique_ele(facets_idx_with_kpts_img0)
        facets_idx_with_kpts_img1, original_idx_img1, seen = find_unique_ele(facets_idx_with_kpts_img1)
        # Extracting the correpondence as per original_idx
        correspondence_after_find_unique_ele = get_correspondence(stacked_corr, facets_idx_with_kpts_img0, facets_idx_with_kpts_img1, config)
        
        ###################################################################################
        return{
            'img0_facetsidx':facets_idx_with_kpts_img0.to(device), 
            'original_idx_img0':original_idx_img0.to(device),
            'img1_facetsidx':facets_idx_with_kpts_img1.to(device), 
            'original_idx_img1':original_idx_img1.to(device),
            #'facets_corr': facets_corr.unsqueeze(0).to(device), 
            'facets_corr': correspondence_after_find_unique_ele, 
            'stacked_corr': stacked_corr.to(device)        
            }
    

    def datapreprocessor(gt, config, device):    
        # Loading the data        
        keypointCoords0 = gt['keypointCoords0']
        keypointCoords1 = gt['keypointCoords1']
        correspondences_1 = torch.as_tensor(gt['correspondences'], dtype=torch.float).to(device)
        correspondences_2 = img2_corr(keypointCoords1, correspondences_1, device)       
        
        # Conversion of keypoint Coordinatess from Spherical to Unit Cartesian coordinates 
        '''I think this is not needed'''
        keypointCoords0 = UnitCartesian(keypointCoords0, device)
        keypointCoords1 = UnitCartesian(keypointCoords1, device)  
        # facets_label
        facet_label_dict = __facet_label(config['nsides'][0], 
                                        keypointCoords0.cpu(), keypointCoords1.cpu(), 
                                        correspondences_1,correspondences_2,  config)

        keypointCoords0 = torch.index_select(keypointCoords0, 0, facet_label_dict['original_idx_img0'])
        keypointCoords1 = torch.index_select(keypointCoords1, 0, facet_label_dict['original_idx_img1'])
        
        keypointDescriptors0 = torch.as_tensor(gt['keypointDescriptors0'], dtype=torch.float).to(device) #.reshape(-1, config['descriptor_dim'])
        keypointDescriptors0 = torch.index_select(keypointDescriptors0, 0, facet_label_dict['original_idx_img0'])
        keypointDescriptors1 = torch.as_tensor(gt['keypointDescriptors1'], dtype=torch.float).to(device)
        keypointDescriptors1 = torch.index_select(keypointDescriptors1, 0, facet_label_dict['original_idx_img1'])
        
        keypointScores0 = torch.as_tensor(gt['keypointScores0'], dtype=torch.float).to(device)
        keypointScores0 = torch.index_select(keypointScores0, 0, facet_label_dict['original_idx_img0'])
        keypointScores1 = torch.as_tensor(gt['keypointScores1'], dtype=torch.float).to(device)
        keypointScores1 = torch.index_select(keypointScores1, 0, facet_label_dict['original_idx_img1'])
        
        scores_corr = torch.as_tensor(gt['scores'], dtype=torch.float).to(device)
        scores_corr = torch.index_select(scores_corr, 0, facet_label_dict['original_idx_img0'])
        
        
        # getting nearest neighbours
        edges1 = knn_graph(keypointCoords0, k=config['knn'], flow= 'target_to_source')        
        edges2 = knn_graph(keypointCoords1, k=config['knn'], flow= 'target_to_source')    # target_to_source

        y_true = {'gt_matches0': correspondences_1.unsqueeze(0), 
                'gt_matches1': correspondences_2.unsqueeze(0)
                }

        data = {'keypointCoords0':keypointCoords0.unsqueeze(0),
                'keypointCoords1':keypointCoords1.unsqueeze(0),
                'keypointDescriptors0':keypointDescriptors0.unsqueeze(0), 
                'keypointDescriptors1':keypointDescriptors1.unsqueeze(0),
                'correspondences': correspondences_1.unsqueeze(0),
                'keypointScores0': keypointScores0.unsqueeze(0),
                'keypointScores1': keypointScores1.unsqueeze(0),
                'edges1':edges1,  
                'edges2':edges2, 
                'scores_corr': scores_corr
                }
        data.update(facet_label_dict)
        
        return data, y_true

    # Data processing and Data loader    ### NPZ
    path = 'HealpixGlue-transfer_learning/data/00000004_00000191.npz'
    g = dict(numpy.load(path))
    #keypointCoords0 = torch.tensor(g['keypointCoords0'])
    #keypointCoords1 = torch.tensor(g['keypointCoords1'])
    #matches0 = torch.tensor(g['correspondences'])

    data, y_true = datapreprocessor(g, default_config, device)
    
    # Visualizing the gt matches and matches after removing the overlapping keypoints for nside 256
    #image0_path = 'image/00000000.jpg'
    #image1_path = 'image/00000001.jpg'
    #keypointCoords0 = data['keypointCoords0'].squeeze().detach().cpu().numpy()
    #keypointCoords1 = data['keypointCoords1'].squeeze().detach().cpu().numpy()
    #matches0 = data['facets_corr'].detach().cpu()
    #plot(matches_between_two_images(image0_path, image1_path, keypointCoords0, keypointCoords1, matches0))
    #exit()
    
    if args.mode == 'train':
        # Loading the model
        matching = SphericalMatching(default_config).to(device) ### base_matcher
        #matching = SphericalMatching_32(default_config).to(device)
        #print(matching)
        #exit()
        # Optimizers
        optimizer = optim.Adam(matching.parameters(), lr=0.0001) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[360], gamma=0.1) # 0.001 -> [132, ]
        loss = []
        # Training the model
        print('Started training')
        best_loss = float('inf')
        
        for i in range(args.epoch):  
            t1 = time.time()
            optimizer.zero_grad(set_to_none=True)
            y_pred, gt_corr, healpix_output = matching(data)  
            #scaling, scaled_loss, y_pred, gt_corr = matching(data)  
            t2 = time.time() 
            y_pred['total_loss'].backward() ### base_matcher
            #scaled_loss.backward()
            t3 = time.time()
            optimizer.step()
            t4 = time.time()
            scheduler.step()
            
            total_loss = y_pred['total_loss'] # epoch_loss/len(train_loader)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(matching.state_dict(), 'HealpixGlue-transfer_learning/saved_model/best_model_parameters_32.pth') # official recommended
            
            # ...log the running loss
            writer.add_scalar('training loss', y_pred['total_loss'].item(), i)
        
            print('\ntime:: model: %.2f, backprop: %.2f, opti: %.4f, total: %.2f' %(t2-t1,t3-t2,t4-t3,t4-t1))
            print('epoch: %s, loss: %.2f, lr: %s' %(i, y_pred['total_loss'].item(), optimizer.param_groups[0]['lr'])) 
            print('Matches', torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item(), y_pred['matches0'].shape[1])
            #print('Matches 32', torch.eq(scaling['indices0'][32], gt_corr['gt_matches0'][32]).sum().item(), scaling['indices0'][32].shape[1])


    if args.mode == 'eval':
        matching_eval = SphericalMatching(default_config).to(device)
        #matching_eval.eval()
        with torch.no_grad():
            # run the model on the test set
            y_pred, gt_corr = matching_eval(data)  
            print('Matches', torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item(), y_pred['matches0'].shape[1])
    
###################################################################################################################################
    
        '''if scaling != None:
            print('Matches 32', torch.eq(scaling['indices0'][32], gt_corr['gt_matches0'][32]).sum().item(), scaling['indices0'][32].shape[1])
            #print('', torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item(), gt_corr['gt_matches0'][256].shape[1])
        if i % 50 == 0:
            print(y_pred['matches0'], gt_corr['gt_matches0'][32].long()) # matches0 -> 16 nside
            
            '''
    
    
    #for param in matching.parameters():
     #   print(param)
    #exit()
    
    
    ### testing
    #matching.eval()
    #with torch.no_grad():
    #    for data in test_loader:
    #        # run the model on the test set
    #        loss, indices1, idx1 = matching(data)
    #        print(indices1[0].unique())



#-------------------------------------------- End --------------------------------------------#

# srun -K -p RTX3090 -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=2 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/mukunda/SM2.sqsh --container-workdir="`pwd`" --pty bash

# python3 -s HealpixGlue-transfer_learning/train_healpixglue_one_img.py
# python3 -s train_healpixglue_one_img.py


################# Debug Matches #
# GT matches work fine
# Data loader has issue: Matches are not refined. SOLVED
# Finding Correspondence:
        ## COG is working fine
        ## Check finding correspondence output