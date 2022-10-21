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


import glob
import torch
from torch.utils.data import Dataset
from utils.Utils import sphericalToCartesian, find_unique_ele
from torch_geometric.nn import knn_graph
import json
import numpy
import healpy as hp


class MyDataset(Dataset):
    def __init__(self, knn, input, config):
        self.knn = knn
        self.config = config
        self.device = config['device']
        self.gt_path = input
        self.data = glob.glob(self.gt_path + "*")
      
    def __len__(self):
        return len(self.data)

    def get_correspondence(self, stacked_corr, facets_idx_with_kpts_img0, facets_idx_with_kpts_img1):
        a,b = torch.split(stacked_corr, 1,1)
        _, idx0 = torch.where(a==facets_idx_with_kpts_img0.to(self.device))
        _, idx1 = torch.where(b==facets_idx_with_kpts_img1.to(self.device))
        corr = -1*torch.ones(facets_idx_with_kpts_img0.shape[0]).to(self.device, dtype=idx1.dtype)
        corr.scatter_(0, idx0, idx1)
        return corr  
    
    def __UnitCartesian(self, points):     
        # Collecting keypoints infocc
        phi, theta =  torch.split(torch.as_tensor(points), 1, dim=1)
        unitCartesian = sphericalToCartesian(phi, theta, 1)
        return unitCartesian.squeeze(2).to(self.device)
    
    def __img2_corr(self, keypointCoords1, correspondences):
        target = -1*torch.ones(len(keypointCoords1))
        for ii, ix_2 in enumerate(correspondences):    
            if ix_2.item() != -1:
                target[int(ix_2.item())]=int(ii)
        return target.to(self.device)

    def __facet_label(self, NSIDE,keypointCoords0, keypointCoords1, img0_corr, img1_corr):        
              
        # Finding in which facet, keypoint is located
        x0,y0,z0 = numpy.split(keypointCoords0.cpu().numpy(),3,1)
        x1,y1,z1 = numpy.split(keypointCoords1.cpu().numpy(),3,1)
        facets_idx_with_kpts_img0 = torch.from_numpy(hp.vec2pix(NSIDE, x0,y0,z0, nest=True)).view(-1)
        facets_idx_with_kpts_img1 = torch.from_numpy(hp.vec2pix(NSIDE, x1,y1,z1, nest=True)).view(-1)
        mask = img0_corr.ge(0)
        kptidx_with_corr_img0 = torch.masked_select(torch.arange(mask.shape[0]).to(self.device), mask).to(torch.int)
        kptidx_with_corr_img1 = torch.masked_select(img0_corr, mask).to(torch.int64)  
        facets_idx_with_kpts_img0_with_corr = torch.index_select(facets_idx_with_kpts_img0.to(self.device), 0, kptidx_with_corr_img0)
        facets_idx_with_kpts_img1_with_corr = torch.index_select(facets_idx_with_kpts_img1.to(self.device), 0, kptidx_with_corr_img1)
        stacked_corr = torch.stack((facets_idx_with_kpts_img0_with_corr, facets_idx_with_kpts_img1_with_corr), 1)
        
        # Fining unique facets and its original index from dataset
        facets_idx_with_kpts_img0, original_idx_img0, seen = find_unique_ele(facets_idx_with_kpts_img0)
        facets_idx_with_kpts_img1, original_idx_img1, seen = find_unique_ele(facets_idx_with_kpts_img1)
        # Extracting the correpondence as per original_idx
        correspondence_after_find_unique_ele = self.get_correspondence(stacked_corr, facets_idx_with_kpts_img0, facets_idx_with_kpts_img1)
        #print(facets_idx_with_kpts_img0)
        ###################################################################################
        return{
            'img0_facetsidx':facets_idx_with_kpts_img0.to(self.device), 
            'original_idx_img0':original_idx_img0.to(self.device),
            'img1_facetsidx':facets_idx_with_kpts_img1.to(self.device), 
            'original_idx_img1':original_idx_img1.to(self.device),
            'facets_corr': correspondence_after_find_unique_ele, 
            'stacked_corr': stacked_corr.to(self.device)        
            }
            
    def datapreprocessor(self, gt):  
        # Loading the data        
        keypointCoords0 = gt['keypointCoords0']
        keypointCoords1 = gt['keypointCoords1']
        correspondences_1 = torch.as_tensor(gt['correspondences'], dtype=torch.float).to(self.device)
        correspondences_2 = self.__img2_corr(keypointCoords1, correspondences_1)       
        
        # Conversion of keypoint Coordinatess from Spherical to Unit Cartesian coordinates 
        keypointCoords0 = self.__UnitCartesian(torch.as_tensor(keypointCoords0))
        keypointCoords1 = self.__UnitCartesian(torch.as_tensor(keypointCoords1)) 
        # facets_label
        facet_label_dict = self.__facet_label(self.config['nsides'][0], 
                                        keypointCoords0.cpu(), keypointCoords1.cpu(), 
                                        correspondences_1,correspondences_2)

        keypointCoords0 = torch.index_select(keypointCoords0, 0, facet_label_dict['original_idx_img0'])
        keypointCoords1 = torch.index_select(keypointCoords1, 0, facet_label_dict['original_idx_img1'])
        
        keypointDescriptors0 = torch.as_tensor(gt['keypointDescriptors0'], dtype=torch.float).to(self.device) #.reshape(-1, config['descriptor_dim'])
        keypointDescriptors0 = torch.index_select(keypointDescriptors0, 0, facet_label_dict['original_idx_img0'])
        keypointDescriptors1 = torch.as_tensor(gt['keypointDescriptors1'], dtype=torch.float).to(self.device)
        keypointDescriptors1 = torch.index_select(keypointDescriptors1, 0, facet_label_dict['original_idx_img1'])
        
        keypointScores0 = torch.as_tensor(gt['keypointScores0'], dtype=torch.float).to(self.device)
        keypointScores0 = torch.index_select(keypointScores0, 0, facet_label_dict['original_idx_img0'])
        keypointScores1 = torch.as_tensor(gt['keypointScores1'], dtype=torch.float).to(self.device)
        keypointScores1 = torch.index_select(keypointScores1, 0, facet_label_dict['original_idx_img1'])
        
        scores_corr = torch.as_tensor(gt['scores'], dtype=torch.float).to(self.device)
        scores_corr = torch.index_select(scores_corr, 0, facet_label_dict['original_idx_img0'])
        
        
        # getting nearest neighbours
        edges1 = knn_graph(keypointCoords0, k=self.config['knn'], flow= 'target_to_source')        
        edges2 = knn_graph(keypointCoords1, k=self.config['knn'], flow= 'target_to_source')    # target_to_source

        y_true = {'gt_matches0': correspondences_1, 
                'gt_matches1': correspondences_2
                }

        data = {'keypointCoords0':keypointCoords0,
                'keypointCoords1':keypointCoords1,
                'keypointDescriptors0':keypointDescriptors0, 
                'keypointDescriptors1':keypointDescriptors1,
                'correspondences': correspondences_1,
                'keypointScores0': keypointScores0,
                'keypointScores1': keypointScores1,
                'edges1':edges1,  
                'edges2':edges2, 
                'scores_corr': scores_corr
                }
        data.update(facet_label_dict)
        
        return data, y_true
    
    def __getitem__(self, idx):
        if self.data[idx].split('.')[-1] == 'json': 
            gt = json.load(open(self.data[idx]))
        if self.data[idx].split('.')[-1] == 'npz': 
            gt = dict(numpy.load(self.data[idx]))
            #print('ok')
        processed_data, y_true = self.datapreprocessor(gt)
        #print('ok')
        #exit()
        processed_data['name'] = self.data[idx].split('/')[-1]
        #print(processed_data['name'])
        return processed_data, y_true