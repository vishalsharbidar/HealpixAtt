import torch
import torch.nn as nn
import healpy as hp
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, knn_graph
import time
from utils.Utils import find_unique_ele


class HealpixAvgPool(nn.AvgPool1d):
    
    def __init__(self, config):
        super().__init__(kernel_size=4)
        #self.val2idx = config['val2idx']
        self.idx2val = config['idx2val']
        self.npix = config['npix']
        self.ipix = config['ipix']
        #self.nested_ipix = config['nested_ipix']
        self.pix2vec = config['pix2vec']
        self.kdtree = config['kdtree']
        self.device = config['device']

    def _find_coordinates(self, nside, u_parent):
        parent_coord = self.pix2vec[nside][u_parent]
        return parent_coord.unsqueeze(0)

        
    def __preprocess_pooling(self, name, NSIDE, facets_with_kpts, descriptor):   
        ''' This function is used to arrange the features of child facets according to the parent facets for pooling

            input:  facets_with_kpts, 
                    descriptor,
                    map_dict_all_facet: val2idx, 
                    map_dict_all_facet_idx2val: idx2val

            output: child_features,
                    uniq_parents_idx,
                    uniq_parents_label,
                    child_facets_for_pooling
        '''
        
        dummy = {0:0}
        print('p',facets_with_kpts.shape)
        map_dict_child_facetlabel_per_keypt = {f:i for i,f in enumerate(facets_with_kpts[0].detach().cpu().numpy())}
        map_dict_child_facetlabel_per_keypt.update(dummy)
        print(len(map_dict_child_facetlabel_per_keypt))
        exit()
        # Finding parent facet for all the child facets
        parent_facet_idx = torch.div(facets_with_kpts[0], 4, rounding_mode='trunc')
        #Finding unique parents and elements original index
        uniq_parents_idx, pooled_idx, seen = find_unique_ele(parent_facet_idx)
        new_desc =  torch.index_select(descriptor, 1, pooled_idx.to(self.device))
        
        # To find facets on next resolution
        nside = NSIDE//2
        # Finding parent facets position
        parent_position = self._find_coordinates(nside, uniq_parents_idx)
        
        # Using unique parents index find the 4 child's indices in healpy nested order 
        child_facets_for_pooling = torch.cat([4*uniq_parents_idx.view(-1, 1) + i for i in range(4)], dim=1) 
        child_facets_for_pooling = child_facets_for_pooling.view(-1).int()
        # Create a mask for facets with keypoints
        child_weights = torch.isin(child_facets_for_pooling, facets_with_kpts.cpu()).float()
        # Removing the facets without keypoints
        filtered_child_facets_for_pooling = torch.mul(child_facets_for_pooling, child_weights)
        
        # Filling the child features 
        features_size = descriptor.shape[2]
        child_features_for_pooling = torch.zeros(uniq_parents_idx.shape[0]*4, features_size)
        
        facet_idx = torch.tensor(list(map(map_dict_child_facetlabel_per_keypt.get, list(filtered_child_facets_for_pooling.cpu().numpy()))))
        child_features_for_pooling = descriptor[0][facet_idx]
        desc_mask = torch.stack([child_weights]*features_size,1).to(self.device)
        child_features_for_pooling = torch.mul(child_features_for_pooling, desc_mask)
      
        return {name + 'child_facets': facets_with_kpts,
                name + 'child_descriptor': descriptor,
                name + 'child_features_for_pooling':child_features_for_pooling.unsqueeze(0), 
                name + 'facetsidx':uniq_parents_idx.unsqueeze(0).to(self.device), 
                name + 'child_facets_for_pooling':child_facets_for_pooling.to(self.device), 
                name + 'filtered_child_facets_for_pooling': filtered_child_facets_for_pooling.to(self.device), 
                name + 'parent_position':parent_position.to(self.device), 
                name + 'pooled_idx': pooled_idx.to(self.device),
                name + 'map_dict_child_facetlabel_per_keypt': map_dict_child_facetlabel_per_keypt}
        

    def forward(self, nside, img0_descriptor, img1_descriptor, data): # , kpts_position, kpts_Scores
        # Preprocessing
        print('img0_facetsidx',data['img0_facetsidx'].shape)
        img0_preprocess_pooling = self.__preprocess_pooling('img0_', nside, data['img0_facetsidx'], img0_descriptor) 
        # Pooling
        img0_child_features = img0_preprocess_pooling['img0_child_features_for_pooling'].permute(0, 2, 1)
        img0_parent_features = F.avg_pool1d(img0_child_features, self.kernel_size)
        img0_preprocess_pooling['img0_parent_features'] = img0_parent_features.permute(0, 2, 1).to(self.device)
        
        # Preprocessing
        img1_preprocess_pooling= self.__preprocess_pooling('img1_', nside, data['img1_facetsidx'], img1_descriptor) # replaced img1_facets_with_kpts -> img1_facetsidx
        # Pooling
        img1_child_features = img1_preprocess_pooling['img1_child_features_for_pooling'].permute(0, 2, 1)
        img1_parent_features = F.avg_pool1d(img1_child_features, self.kernel_size)
        img1_preprocess_pooling['img1_parent_features'] = img1_parent_features.permute(0, 2, 1).to(self.device)
        
        img0_preprocess_pooling.update(img1_preprocess_pooling)  
        return img0_preprocess_pooling


      