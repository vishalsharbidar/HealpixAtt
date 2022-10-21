import json
import torch
import torch.nn as nn
import healpy as hp
from sklearn.neighbors import KDTree
import healpy as hp
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, knn_graph, GATConv
from pooling.healpixAvgPool import HealpixAvgPool
from torch.nn import ReLU

# Does Spherical Chebyshev convolution
class SphericalChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, aggr):
        super().__init__()
        #self.conv = ChebConv(in_channels, out_channels, K=K, aggr=aggr)
        self.conv = GATConv(in_channels, out_channels, heads=1, aggr=aggr)
        
    def forward(self, x, position):
        edges = knn_graph(position, k=20, flow= 'target_to_source')
        #print('edges', x.shape, edges.shape)
        #x = self.conv(x, edges)
        x = self.conv(x.squeeze(), edges)
        return x.unsqueeze(0)


# Does Spherical Pooling using HealpixAvgPool
class SphericalPooling(nn.AvgPool1d):
    def __init__(self, config):
        super().__init__(kernel_size=4)
        self.config = config
        self.pool = HealpixAvgPool(self.config)        
    
    def forward(self, nside, img0_descriptor, img1_descriptor, data):
        pooled_output = self.pool(nside, img0_descriptor, img1_descriptor, data)
        return pooled_output, pooled_output['img0_parent_features'], pooled_output['img1_parent_features']    


# Does Multi Scale Pooling
class HealpixHierarchy(nn.Module):
    def __init__(self, in_channels, out_channels, K, aggr, config):
        super(HealpixHierarchy, self).__init__()
        self.config = config
        self.conv1 = SphericalChebConv(in_channels, 256, K=K, aggr=aggr) # 256
        self.conv2 = SphericalChebConv(256, 256, K=K, aggr=aggr) # 128
        self.conv3 = SphericalChebConv(256, 512, K=K, aggr=aggr) # 64
        self.conv4 = SphericalChebConv(512, out_channels, K=K, aggr=aggr) # 32
        #self.conv5 = SphericalChebConv(1024, out_channels, K=K, aggr=aggr)
        self.pool = SphericalPooling(self.config)
        
    def forward(self, x0, x1, data): #Nside, facets_label, descriptor
        output_ = {}
        x0_ = {}
        x1_ = {}
        #print(x0.shape, data['keypointCoords0'].squeeze(0).shape)
        #exit()
        x0 = self.conv1(x0, data['keypointCoords0'].squeeze(0)) 
        x1 = self.conv1(x1, data['keypointCoords1'].squeeze(0))
        NSIDE = self.config['nsides'][0]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0, x1, data)
                
        x0 = self.conv2(x0_[NSIDE], output_[NSIDE]['img0_parent_position'].squeeze(0))
        x1 = self.conv2(x1_[NSIDE], output_[NSIDE]['img1_parent_position'].squeeze(0))
        NSIDE = self.config['nsides'][1]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0, x1, output_[NSIDE*2])
        
        x0 = self.conv3(x0_[NSIDE], output_[NSIDE]['img0_parent_position'].squeeze(0))
        x1 = self.conv3(x1_[NSIDE], output_[NSIDE]['img1_parent_position'].squeeze(0))
        NSIDE = self.config['nsides'][2]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0, x1, output_[NSIDE*2])

        x0 = self.conv4(x0_[NSIDE], output_[NSIDE]['img0_parent_position'].squeeze(0))
        x1 = self.conv4(x1_[NSIDE], output_[NSIDE]['img1_parent_position'].squeeze(0))
        NSIDE = self.config['nsides'][3]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0, x1, output_[NSIDE*2])
        
        return output_, x0_, x1_