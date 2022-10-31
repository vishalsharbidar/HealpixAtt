import json
import torch
import torch.nn as nn
import healpy as hp
from sklearn.neighbors import KDTree
import healpy as hp
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, knn_graph, GATConv
from torch_geometric.utils import to_dense_adj
from pooling.healpixAvgPool import HealpixAvgPool
from torch.nn import ReLU
from model.superglue import AttentionalPropagation
from model.masked_SelfAttention import SelfAttentionalGNN
from typing import List, Tuple

# Does Spherical Chebyshev convolution
class SphericalChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, aggr):
        super().__init__()
        #self.conv = ChebConv(in_channels, out_channels, K=K, aggr=aggr)
        self.conv = GATConv(in_channels, out_channels, heads=1, aggr=aggr)
        
    def forward(self, x, position):
        edges = knn_graph(position, k=80, flow= 'target_to_source')
        x = self.conv(x, edges)
        return x

def knn(position):
    edges = knn_graph(position.squeeze(0), k=80, flow= 'target_to_source')
    return edges

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
        self.attn1 = SelfAttentionalGNN(feature_dim=in_channels, layer_names=['self'])
        self.attn2 = SelfAttentionalGNN(feature_dim=in_channels, layer_names=['self'])
        self.attn3 = SelfAttentionalGNN(feature_dim=in_channels, layer_names=['self'])
        self.attn4 = SelfAttentionalGNN(feature_dim=in_channels, layer_names=['self'])

        self.pool = SphericalPooling(self.config)
        
    def forward(self, x0, x1, data): #Nside, facets_label, descriptor
        output_ = {}
        x0_ = {}
        x1_ = {}
        
        #print(x0.shape, data['edges1'].shape, x1.shape, data['edges2'].shape)
        #exit()

        x0, x1 = self.attn1(x0, data['edges1'].squeeze(), x1, data['edges2'].squeeze())
        print('desc',x0.shape, x1.shape)
        NSIDE = self.config['nsides'][0]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0.transpose(2,1), x1.transpose(2,1), data)
        print(NSIDE, x0_[NSIDE].shape,  x1_[NSIDE].shape, output_[NSIDE]['img0_parent_position'].shape)
        exit()
        edges0, edges1 = knn(output_[NSIDE]['img0_parent_position']), knn(output_[NSIDE]['img1_parent_position'])     
        x0, x1 = self.attn2(x0_[NSIDE].transpose(2,1), edges0, x1_[NSIDE].transpose(2,1), edges1)
        NSIDE = self.config['nsides'][1]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0.transpose(2,1), x1.transpose(2,1), output_[NSIDE*2])
        #print(NSIDE, x0_[NSIDE].shape,  x1_[NSIDE].shape, output_[NSIDE]['img0_parent_position'].shape)
        
        edges0, edges1 = knn(output_[NSIDE]['img0_parent_position']), knn(output_[NSIDE]['img1_parent_position'])     
        x0, x1 = self.attn3(x0_[NSIDE].transpose(2,1), edges0, x1_[NSIDE].transpose(2,1), edges1)
        NSIDE = self.config['nsides'][2]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0.transpose(2,1), x1.transpose(2,1), output_[NSIDE*2])
        #print(NSIDE, x0_[NSIDE].shape,  x1_[NSIDE].shape, output_[NSIDE]['img0_parent_position'].shape)


        edges0, edges1 = knn(output_[NSIDE]['img0_parent_position']), knn(output_[NSIDE]['img1_parent_position'])     
        x0, x1 = self.attn4(x0_[NSIDE].transpose(2,1), edges0, x1_[NSIDE].transpose(2,1), edges1)
        NSIDE = self.config['nsides'][3]
        output_[NSIDE], x0_[NSIDE], x1_[NSIDE]  = self.pool(NSIDE, x0.transpose(2,1), x1.transpose(2,1), output_[NSIDE*2])
        #print(NSIDE, x0_[NSIDE].shape,  x1_[NSIDE].shape, output_[NSIDE]['img0_parent_position'].shape)
        #exit()

        return output_, x0_, x1_