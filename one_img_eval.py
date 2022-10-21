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
from model.matcher import SphericalMatching
from model.loss import criterion
import json
from utils.Utils import sphericalToCartesian, find_unique_ele
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
    parser.add_argument('--epoch', type=int, default=1000,
        help=' Number of epochs', metavar='')
    parser.add_argument('--descriptor_dim', type=int, default=128,
        help=' Dimension of descriptor', metavar='') # 64, 128, 256
    parser.add_argument('--sinkhorn', type=int, default=500,
        help=' Sinkhorn iterations', metavar='')
    parser.add_argument('--GNN_layers', type=str, nargs='+', default=['cross'],
        help=' GNN layers', metavar='') # ['self', 'cross']
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
                }

    default_config.update(healpy_config)

    ###########################################    

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
    
        mask = img0_corr.ge(0)
        healpix_kdtree =  config['kdtree'][NSIDE]
        device = config['device']       
        # Finding in which facet, keypoint is located
        x0,y0,z0 = numpy.split(keypointCoords0.cpu().numpy(),3,1)
        x1,y1,z1 = numpy.split(keypointCoords1.cpu().numpy(),3,1)
        facets_idx_with_kpts_img0 = torch.from_numpy(hp.vec2pix(NSIDE, x0,y0,z0, nest=True)).view(-1)
        facets_idx_with_kpts_img1 = torch.from_numpy(hp.vec2pix(NSIDE, x1,y1,z1, nest=True)).view(-1)
        # Fining unique facets and its original index from dataset
        facets_idx_with_kpts_img0, original_idx_img0, seen = find_unique_ele(facets_idx_with_kpts_img0)
        facets_idx_with_kpts_img1, original_idx_img1, seen = find_unique_ele(facets_idx_with_kpts_img1)
        # Extracting the correpondence as per original_idx
        img0_new_corr = img0_corr[original_idx_img0].long()
        #img1_new_corr = img1_corr[original_idx_img1].long()
        
        # Finding correspondence between facets of img0 and img1
        facets_corr = -1*torch.ones(img0_new_corr.shape[0])
        for ix, img1_kidx in enumerate(img0_new_corr):
            if img1_kidx.item()!=-1:
                if (original_idx_img1==img1_kidx.item()).nonzero().shape[0] == 0:
                    pass
                else:
                    facets_corr[ix] =(original_idx_img1==img1_kidx.item()).nonzero().item()

        
        return{
            'img0_facetsidx':facets_idx_with_kpts_img0.to(device), 
            'original_idx_img0':original_idx_img0.to(device),
            'img1_facetsidx':facets_idx_with_kpts_img1.to(device), 
            'original_idx_img1':original_idx_img1.to(device),
            'facets_corr': facets_corr.unsqueeze(0).to(device)            
            }
    
    def datapreprocessor(gt, config, device):    
        # Loading the data        
        keypointCoords0 = gt['keypointCoords0']
        keypointCoords1 = gt['keypointCoords1']
        correspondences_1 = torch.as_tensor(gt['correspondences'], dtype=torch.float).to(device)
        correspondences_2 = img2_corr(keypointCoords1, correspondences_1, device)       
        # Conversion of keypoint Coordinatess from Spherical to Unit Cartesian coordinates
        keypointCoords0 = UnitCartesian(keypointCoords0, device)
        keypointCoords1 = UnitCartesian(keypointCoords1, device)  
        # facets_label
        facet_label_dict = __facet_label(config['nsides'][0], 
                                        keypointCoords0.cpu(), keypointCoords1.cpu(), 
                                        correspondences_1,correspondences_2,  config)

        
        #facet_label_dict['original_idx_img1']
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

    # Data processing and Data loader
    #dataset = MyDataset(args.knn, args.input, device)
    #train_set, test_set = random_split(dataset, [2, 1])

    #train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    #test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    # Data processing and Data loader for single image
    ### JSON
    #path = 'HealpixGlue/data/2k_00000014_00000015.json'
    #g = json.load(open(path))
    ### NPZ
    path = 'HealpixGlue/data/00000000_00000001.npz'
    g = dict(numpy.load(path))
    data, y_true = datapreprocessor(g, default_config, device)

    # Loading the model
    matching = SphericalMatching(default_config).to(device)
    print("Model state_dict:")
    for param_tensor in matching.state_dict():
        print(param_tensor, "\t", matching.state_dict()[param_tensor].size())
        
    #print(matching)
    exit()

    # Optimizers
    optimizer = optim.Adam(matching.parameters(), lr=0.0001) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[360], gamma=0.1) # 0.001 -> [132, ]
    loss = []
    # Training the model
    print('Started training')
    best_loss = float('inf')
    
    ### testing
    matching.load_state_dict(torch.load('HealpixGlue/weights/best_model_parameters.pth'))
    matching.eval()
    with torch.no_grad():
        # run the model on the test set
        y_pred, gt_corr = matching(data)  
        print('Matches', torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item(), y_pred['matches0'].shape[1])



#-------------------------------------------- End --------------------------------------------#

# srun -K -p RTX3090 -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=2 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/mukunda/SM2.sqsh --container-workdir="`pwd`" --pty bash

# python3 -s HealpixGlue/one_img_eval.py
