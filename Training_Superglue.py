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
from utils.mydataset import MyDataset
import time
from utils.FacetsMapping import HealpyFacetsMappingDict
import healpy as hp
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SphericalMatching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, nargs='+', default='/netscratch/mukunda/npzdata150/data/',
        help=' Input path', metavar='')
    parser.add_argument('--nsides', nargs='+', type=float, default= [256, 128, 64, 32, 16],
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
    parser.add_argument('--epoch', type=int, default=500,
        help=' Number of epochs', metavar='')
    parser.add_argument('--descriptor_dim', type=int, default=128,
        help=' Dimension of descriptor', metavar='') # 64, 128, 256
    parser.add_argument('--batch_size', type=int, default=4,
        help=' Batch size of training images', metavar='')
    parser.add_argument('--sinkhorn', type=int, default=100,
        help=' Sinkhorn iterations', metavar='')
    parser.add_argument('--GNN_layers', type=str, nargs='+', default=['cross'],
        help=' GNN layers', metavar='') # ['self', 'cross']
    parser.add_argument('--conv', type=str, nargs='+', default='GAT', # GAT, Cheb
        help=' Model mode', metavar='') 
    parser.add_argument('--mode', type=str, nargs='+', default='train', # ['train', 'test', 'single_image']
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
    
    folder = args.input.split('/')[-2]
    #writer = SummaryWriter('tb_path/HealpixGlue/'+str(folder)+'/')
    writer = SummaryWriter('tb_path/HealpixGlue/lr001/')

    # Healpy Facets Mapping 
    print('Caching Healpy Facets Mapping')
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
                    'conv' : args.conv
                }

    default_config.update(healpy_config)

    # Data processing and Data loader
    dataset = MyDataset(args.knn, args.input, default_config)
    train, test_set = random_split(dataset, [132,20])
    train_set, val_set = random_split(train, [112,20])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    
    
    if args.mode == 'train':
        # Loading the model
        matching = SphericalMatching(default_config).to(device)
        # Optimizers
        optimizer = optim.Adam(matching.parameters(), lr=0.0001*args.batch_size, weight_decay=1e-5) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
        # Training the model
        print('Training begins')
        min_valid_loss = float('inf')
        total_loss = 0  
        gradient_accumulations = 16
        for i in range(args.epoch):  
            # Training
            total_loss = 0  
            t1 = time.time()
            for data_idx, data_ in enumerate(train_loader):
                data, y_true = data_
                optimizer.zero_grad(set_to_none=True)
                # Forward pass
                y_pred, gt_corr, output = matching(data)  
                t2 = time.time()
                # Calculate gradients
                (y_pred['total_loss'] / gradient_accumulations).backward()
                #scaled_loss.backward()
                t3 = time.time()
                # Update Weights
                if (data_idx + 1) % gradient_accumulations == 0:
                    optimizer.step()
                    matching.zero_grad()
                t4 = time.time()
                total_loss += y_pred['total_loss']/args.batch_size # epoch_loss/len(train_loader)

            # Validation
            valid_loss = 0.0
            matching.eval()     
            for data, y_true in valid_loader:                
                # Forward Pass
                y_pred, gt_corr, output = matching(data) 
                # Calculate Loss
                valid_loss += y_pred['total_loss']/args.batch_size

            #scheduler.step()

            print(f'\nEpoch {i+1} \t\t Training Loss: {total_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
            # Saving the best model
            if min_valid_loss > valid_loss/len(valid_loader):
                print(f'\t\t\t Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss/len(valid_loader):.6f}) \t Saving The Model')
                min_valid_loss = valid_loss/len(valid_loader)
                torch.save(matching.state_dict(), 'HealpixAtt/saved_model/'+str(folder)+'/best_model_parameters.pth') # official recommended
            
            # log the running loss
            writer.add_scalar('loss/training', total_loss/len(train_loader), i)
            writer.add_scalar('loss/validation', valid_loss/len(valid_loader), i)
            
            print('Healpix Matches', torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item(), y_pred['matches0'].shape[1])
            t2 = time.time()
            print('epoch time: %.2f' %(t2-t1))

    if args.mode == 'test':
        ### testing
        matching_test = SphericalMatching(default_config).to(device)
        with torch.no_grad():
            for data, y_true in test_loader:
                # run the model on the test set
                y_pred, gt_corr, output  = matching_test(data) 
                print('%s: %s' %(str(data['name']),torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item()))

                output_path = 'HealpixAtt/output/'+str(folder)+'/'+ str(data['name'][0]) 
                out_file = open(output_path, "w")
                out = y_pred['matches0'].detach().tolist()
                json.dump(out, out_file)
                out_file.close()
                

    if args.mode == 'single_image':
        data, y_true = next(iter(train_loader))
        # Loading the model
        matching = SphericalMatching(default_config).to(device)
        #print(matching)
        #exit()
        # Optimizers
        optimizer = optim.Adam(matching.parameters(), lr=0.0001*args.batch_size) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[260], gamma=0.1)
        # Training the model
        print('Training begins')
        min_valid_loss = float('inf')
        total_loss = 0 
        for i in range(args.epoch):  
            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            y_pred, gt_corr, output = matching(data)  
            t2 = time.time()
            # Calculate gradients
            y_pred['total_loss'].backward()
            #scaled_loss.backward()
            t3 = time.time()
            # Update Weights
            optimizer.step()
            t4 = time.time()
            loss = y_pred['total_loss']
            scheduler.step()
            print(f'\nEpoch {i+1} \t\t Training Loss: {loss}')
            writer.add_scalar('Single_img_loss/training', loss, i)
            print('%s matches out of %s' %(torch.eq(y_pred['matches0'], gt_corr['gt_matches0'][16]).sum().item(), gt_corr['gt_matches0'][16].shape[1]))
#-------------------------------------------- End --------------------------------------------#

# srun -K --partition=A100 --nodes=1 --ntasks=4 --cpus-per-task=2 --gpus-per-task=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/mukunda/SM2.sqsh --container-workdir="`pwd`" --pty bash
# srun -K -p RTX3090 -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=2 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/mukunda/keops.sqsh --container-workdir="`pwd`" --pty bash

# python3 -s HealpixAtt/Training_Superglue.py
