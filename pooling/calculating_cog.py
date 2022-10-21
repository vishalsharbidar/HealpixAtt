
import torch
import torch.nn as nn
import time
import healpy as hp

class Center_of_Gravity(nn.Module):
    ''' Calculate the Center of gravity of child factes which has keypoints.
        input:  keypoints position in unitcartesian
                keypoints scores
        output: new coordinates for correspondence after pooling
                new score 
        '''
    def __init__(self):
        super().__init__()
    
    def forward(self, position, scores):
        x,y,z = torch.split(position, 1, dim=1)
        #print(x,y,z)
        scores = scores.view(-1,4).T
        mask = scores.greater(0).sum(dim=0)

        # To find the center of gravity  
        x_coordinate = torch.nan_to_num(torch.div(torch.sum(torch.mul(scores, x.view(-1,4).T),0), torch.sum(scores,0)))
        y_coordinate = torch.nan_to_num(torch.div(torch.sum(torch.mul(scores, y.view(-1,4).T),0), torch.sum(scores,0)))
        z_coordinate = torch.nan_to_num(torch.div(torch.sum(torch.mul(scores, z.view(-1,4).T),0), torch.sum(scores,0)))
        
        #print('x_coordinate\n',x_coordinate)
        position_cog = torch.stack((x_coordinate, y_coordinate, z_coordinate), dim=1)
        score_cog = torch.nan_to_num(torch.div(torch.sum(scores,0), mask))
        return position_cog, score_cog 



class Calculating_COG(nn.Module):
    ''' 
        This function is used to calculate parent position and score using Center of gravity. 
        Function uses __center_of_Gravity to find parent position and score using Center of gravity 
        and __correspondence_pooled to find the correspondence after pooling.
            input:  nside,
                    correspondences,
                    facets_with_kpts_img0,
                    facets_with_kpts_img1,
                    child_facets_for_pooling_img0,
                    uniq_parents_img0,
                    kpts_unitCartesian1,
                    kpts_Scores1

            output: parent_position_cog, 
                    parent_scores_cog
        '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.COG = Center_of_Gravity()

    def forward(self, nside, correspondences, pooling_output, keypointScores1):
        t1 = time.time()
        # creating a mask using groundtruth
        mask = correspondences.ge(0)
        
        # Using mask find indices of keypoints with correspondence for image 0 and image 1
        kptidx_with_corr_img0 = torch.masked_select(torch.arange(mask.shape[1]).to(self.config['device']), mask).to(torch.int)
        kptidx_with_corr_img1 = torch.masked_select(correspondences, mask).to(torch.int)  

        # Select the facets with correspondence using the indices 
        facets_with_corr_img0 = torch.index_select(pooling_output['img0_child_facets'].squeeze(), 0, kptidx_with_corr_img0).squeeze()      
        facets_with_corr_img1 = torch.index_select(pooling_output['img1_child_facets'].squeeze(), 0, kptidx_with_corr_img1).squeeze()
        
        
        # Mapping the facets from image0 with image1
        dummy = {0:0}
        img0_for_dict = facets_with_corr_img0.detach().cpu().numpy()
        img1_for_dict = facets_with_corr_img1.detach().cpu().numpy()
        map_dict_facets_corr = {img0_for_dict[ele] : img1_for_dict[ele] for ele in range(len(facets_with_corr_img0))}
        map_dict_facets_corr.update(dummy)
        
        # maps all the keypoints with its nearest facets image 1
        map_dict_child_facetlabel_per_keypt_img1 = {f:i for i,f in enumerate(pooling_output['img1_child_facets'][0].detach().cpu().numpy())}
        map_dict_child_facetlabel_per_keypt_img1.update(dummy)
    
        # Creating a mask to find facets with correspondence 
        mask_facets_with_corr_img0 = torch.isin(pooling_output['img0_filtered_child_facets_for_pooling'], facets_with_corr_img0).float() 
        
        ##### Removing the facets without correspondence
        facets_with_corr_for_pooling_img0 = torch.mul(pooling_output['img0_filtered_child_facets_for_pooling'], mask_facets_with_corr_img0).int() 
        
        # Calculating childs position for COG
        a = torch.tensor(list(map(map_dict_facets_corr.get, list(facets_with_corr_for_pooling_img0.cpu().numpy()))))
        x,y,z = hp.pix2vec(nside, a, nest=True)
        img1_child_position_for_cog = torch.stack([x,y,z], 1).to(self.config['device'])
        pos_mask = torch.stack([mask_facets_with_corr_img0]*3,1)
        img1_child_position_for_cog = torch.mul(img1_child_position_for_cog, pos_mask)

        # Calculating childs score for COG
        b = torch.tensor(list(map(map_dict_child_facetlabel_per_keypt_img1.get, list(a.numpy()))))
        img1_child_scores_for_cog = torch.mul(keypointScores1[0][b], mask_facets_with_corr_img0)
        
        # Calculating the Center of gravity for img 1
        parent_position_cog, parent_scores_cog = self.COG(img1_child_position_for_cog, img1_child_scores_for_cog)
        
        t2 = time.time()
        
        return {'img1_parent_position_cog': parent_position_cog, 
                'img1_parent_scores_cog': parent_scores_cog.unsqueeze(0), 
                'kptidx_with_corr_img0': kptidx_with_corr_img0, 
                'kptidx_with_corr_img1': kptidx_with_corr_img1,
                'img1_keypointScores':keypointScores1 # Not used anywhere and I think not needed, might become a bug
                }

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    cog = Center_of_Gravity().to(device)
    position = torch.rand(8,3)
    score = torch.rand(8)
    s = time.time()
    p, score = cog(position, score)    
    e = time.time()
    #print(e-s)
    #print(p.shape, score.shape)

#### To run
# python3 -s Healpixglue/pooling/center_of_gravity.py