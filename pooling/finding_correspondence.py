import torch
import torch.nn as nn
import time
from pooling.calculating_cog import Calculating_COG
from sklearn.neighbors import KDTree
from utils.ploting import matches_between_two_images, plot, keypoints_on_healpix_map_over_Single_image
import healpy as hp

def get_correspondence(stacked_corr, facets_idx_with_kpts_img0, facets_idx_with_kpts_img1, config):
    a,b = torch.split(stacked_corr, 1,1)
    _, idx0 = torch.where(a==facets_idx_with_kpts_img0.to(config['device']))
    _, idx1 = torch.where(b==facets_idx_with_kpts_img1.to(config['device']))
    corr = -1*torch.ones(facets_idx_with_kpts_img0.shape[0]).to(config['device'], dtype=idx1.dtype)
    corr.scatter_(0, idx0, idx1)
    return corr  

def find_facet_correspondence(facets_corr, img0_pooled_idx, img1_pooled_idx, device):
    img0_new_corr = torch.index_select(facets_corr, 1, img0_pooled_idx).squeeze()
    # Finding correspondence for between facets
    facets_corr = -1*torch.ones(img0_new_corr.shape[0]).to(device)
    for ix, img1_corr in enumerate(img0_new_corr):
        if img1_corr.item()!=-1:
            if (img1_pooled_idx==img1_corr.item()).nonzero().shape[0] == 0:
                pass
            else:
                facets_corr[ix] =(img1_pooled_idx==img1_corr.item()).nonzero().item()
    return facets_corr.unsqueeze(0)

def img2_corr(keypointCoords1, correspondences, device):
    target = -1*torch.ones(len(keypointCoords1))
    for ii, ix_2 in enumerate(correspondences):    
        if ix_2.item() != -1:
            target[int(ix_2.item())]=int(ii)
    return target.to(device)

####----------------------------------------Finding_Correspondence-------------------------------------------------------------------####

class Finding_Correspondence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.calculating_COG = Calculating_COG(self.config)
    
    def new_corr(self, nside, facets_corr, img1_facetsidx, keypointScores1, hh_output):
        # Select the info for given nside
        pooling_output = hh_output[nside]
        # Finding COG 
        COG_output= self.calculating_COG(nside, facets_corr, pooling_output, keypointScores1) #pooling_output['facets_corr']
        img1_parent_position_cog = COG_output['img1_parent_position_cog'].unsqueeze(0).to(self.config['device'])
        
        ############################### To Visualize Put below code here #######################################
        
        # To find the correspondence for different resolution.
        pooling_output['facets_corr'] = find_facet_correspondence(facets_corr, pooling_output['img0_pooled_idx'],
                                                pooling_output['img1_pooled_idx'], self.config['device'])
     
        mask = pooling_output['facets_corr'].ge(0)
        kptidx_with_corr_img0 = torch.masked_select(torch.arange(mask.shape[1]).to(self.config['device']), mask).to(torch.int)
        kptidx_with_corr_img1 = torch.masked_select(pooling_output['facets_corr'], mask).to(torch.int64)  
        
        img1_parent_position_cog_with_corr = torch.index_select(img1_parent_position_cog, 1, kptidx_with_corr_img0)
        # For defined Nside calculating healpix facets cartesian coordinates
        NSIDE = nside//2        
        healpix_kdtree = self.config['kdtree'][NSIDE]

        # Finding nearest neighbours facets for image 0 keypoints
        x0,y0,z0 = torch.split(img1_parent_position_cog_with_corr[0],1,1) # need to work on this
        facets_idx_with_corr = torch.from_numpy(hp.vec2pix(NSIDE, x0.detach().cpu().numpy(),y0.detach().cpu().numpy(),z0.detach().cpu().numpy(), nest=True)).view(-1)
        #facets_idx_with_corr = torch.tensor(healpix_kdtree.query(img1_parent_position_cog_with_corr[0].cpu(), k=1, return_distance=False)).squeeze().to(self.config['device'])              
        
        pooled_output = pooling_output['img1_facetsidx'][0].clone()
        pooling_output['img1_facetsidx_new'] = pooled_output.scatter_(0, kptidx_with_corr_img1, facets_idx_with_corr.to(self.config['device']))
        return pooling_output, COG_output

    def forward(self, output, data):
        gt = {}
        gt['img1_facetsidx_new'] = {}
        gt['gt_matches0'] = {}
        gt['gt_matches1'] = {}

        NSIDE = self.config['nsides'][0]
        gt['img1_facetsidx_new'][NSIDE] = data['img1_facetsidx']
        gt['gt_matches0'][NSIDE] = data['facets_corr'].int()
        gt['gt_matches1'][NSIDE] = img2_corr(data['img1_facetsidx'].squeeze(0), gt['gt_matches0'][NSIDE].squeeze(0), self.config['device']).int()
        #print('gt',NSIDE, gt['gt_matches0'][NSIDE].shape, gt['gt_matches1'][NSIDE].shape, data['img1_facetsidx'].shape)
        
        # To Find updated target image facets after doing COG and calculates new correspondence for given resolution  
        out_256, cog_256 = self.new_corr(NSIDE, data['facets_corr'], data['img1_facetsidx'], data['keypointScores1'], output)     

        NSIDE = self.config['nsides'][1]
        gt['img1_facetsidx_new'][NSIDE] = out_256['img1_facetsidx_new']
        gt['gt_matches0'][NSIDE] = out_256['facets_corr'].int()
        gt['gt_matches1'][NSIDE] = img2_corr(out_256['img1_facetsidx_new'].squeeze(0), gt['gt_matches0'][NSIDE].squeeze(0), self.config['device']).int()
        #print('gt',NSIDE, gt['gt_matches0'][NSIDE].shape, gt['gt_matches1'][NSIDE].shape)
        
        # To Find updated target image facets after doing COG and calculates new correspondence for given resolution  
        out_128, cog_128 = self.new_corr(NSIDE, out_256['facets_corr'], out_256['img1_facetsidx'], cog_256['img1_keypointScores'], output)
        
        NSIDE = self.config['nsides'][2]
        gt['img1_facetsidx_new'][NSIDE] = out_128['img1_facetsidx_new']
        gt['gt_matches0'][NSIDE] = out_128['facets_corr'].int()
        gt['gt_matches1'][NSIDE] = img2_corr(out_128['img1_facetsidx_new'].squeeze(0), gt['gt_matches0'][NSIDE].squeeze(0), self.config['device']).int()
        #print('gt',NSIDE, gt['gt_matches0'][NSIDE].shape, gt['gt_matches1'][NSIDE].shape)
        
        # To Find updated target image facets after doing COG and calculates new correspondence for given resolution        
        out_64, cog_64 = self.new_corr(NSIDE, out_128['facets_corr'], out_128['img1_facetsidx'], cog_128['img1_keypointScores'], output)
        
        NSIDE = self.config['nsides'][3]
        gt['img1_facetsidx_new'][NSIDE] = out_64['img1_facetsidx_new']
        gt['gt_matches0'][NSIDE] = out_64['facets_corr'].int()
        gt['gt_matches1'][NSIDE] = img2_corr(out_64['img1_facetsidx_new'].squeeze(0), gt['gt_matches0'][NSIDE].squeeze(0), self.config['device']).int()
        #print('gt',NSIDE, gt['gt_matches0'][NSIDE].shape, gt['gt_matches1'][NSIDE].shape)
        
        # To Find updated target image facets after doing COG and calculates new correspondence for given resolution  
        out_32, cog_32 = self.new_corr(NSIDE, out_64['facets_corr'], out_64['img1_facetsidx'], cog_64['img1_keypointScores'], output)
        
        NSIDE = self.config['nsides'][4]
        gt['img1_facetsidx_new'][NSIDE] = out_32['img1_facetsidx_new']
        gt['gt_matches0'][NSIDE] = out_32['facets_corr'].int()
        gt['gt_matches1'][NSIDE] = img2_corr(out_32['img1_facetsidx_new'].squeeze(0), gt['gt_matches0'][NSIDE].squeeze(0), self.config['device']).int()
        #print('gt',NSIDE, gt['gt_matches0'][NSIDE].shape, gt['gt_matches1'][NSIDE].shape)
      
        # To visualize
        t, p = hp.pix2ang(64, out_128['img0_facetsidx'].detach().cpu(), nest=True)
        img0 = torch.stack((t, p), 1)
        t1, p1 = hp.pix2ang(64, out_128['img1_facetsidx_new'].detach().cpu(), nest=True)        
        img1 = torch.stack((t1, p1), 1)
        
        ###############################  Visualization #######################################
        # Visualizing the gt matches and matches after removing the overlapping keypoints for nside 256
        #image0_path = 'image/00000004.jpg'
        #image1_path = 'image/00000191.jpg'
        #keypointCoords0 = img0 #data['keypointCoords0'].squeeze().detach().cpu().numpy()
        #keypointCoords1 = img1
        #matches0 = out_128['facets_corr'][0].detach().cpu()
        #plot(matches_between_two_images(image0_path, image1_path, keypointCoords0, keypointCoords1, matches0))
        #exit()
        
        return gt

###############################  Visualization #######################################
'''################### Matching ###############################
cog_corr = -1*torch.ones(COG_output['img1_parent_position_cog'].shape[0])
#print(COG_output['img1_parent_position_cog'].shape)
#exit()

for ix, ele in enumerate(COG_output['img1_parent_position_cog'].detach().cpu().numpy()):
    if ele.any()>0:
        cog_corr[ix] = hp.vec2pix(128, ele[0], ele[1], ele[2], nest=True)

#print(facets_corr, cog_corr.long())
# 
mask = cog_corr.ge(0).to(self.config['device'])
kptidx_with_corr_img0 = torch.masked_select(torch.arange(mask.shape[0]).to(self.config['device']), mask).to(torch.int)

facets_with_corr_img0 = torch.index_select(pooling_output['img0_facetsidx'], 0, kptidx_with_corr_img0).squeeze()      
facets_with_corr_img1_cog = torch.index_select(cog_corr.to(self.config['device']), 0, kptidx_with_corr_img0).squeeze()      
stacked_corr = torch.stack((facets_with_corr_img0, facets_with_corr_img1_cog), 1)
#print(stacked_corr.shape, facets_with_corr_img0, facets_with_corr_img1_cog)
print(facets_with_corr_img0.detach().cpu(), facets_with_corr_img1_cog.detach().cpu().int())

#ipix = pooling_output['img0_facetsidx'].detach().cpu()
t, p = hp.pix2ang(128, facets_with_corr_img0.detach().cpu(), nest=True)
img0 = torch.stack((t, p), 1)

#ipix1 = pooling_output['img1_child_facets'].detach().cpu()
t1, p1 = hp.pix2ang(128, facets_with_corr_img1_cog.detach().cpu().int(), nest=True)
img1 = torch.stack((t1, p1), 1)
print(img0.shape, img1.shape)
#exit()
# Visualizing the gt matches and matches after removing the overlapping keypoints for nside 256
image0_path = 'image/00000000.jpg'
image1_path = 'image/00000001.jpg'
keypointCoords0 = img0 #data['keypointCoords0'].squeeze().detach().cpu().numpy()
keypointCoords1 = img1  #data['keypointCoords1'].squeeze().detach().cpu().numpy()
matches0 = torch.arange(facets_with_corr_img0.shape[0])
#matches0 = facets_corr.detach().cpu()
plot(matches_between_two_images(image0_path, image1_path, keypointCoords0, keypointCoords1, matches0))
exit()

################### END ###############################
'''