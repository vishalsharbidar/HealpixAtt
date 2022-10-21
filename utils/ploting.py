import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.nn import knn_graph
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2


#### To plot Matches between two image
def plot(img):    
    figure(figsize=(20, 20), dpi=160)
    plt.imshow(img)
    plt.show()

    
def Spherical_to_Pixel_Coord(phi,theta, imgWidth, imgHeight):
    KPi2Inverted= 1/(2*np.pi)
    KPiInverted = 1/(np.pi)                          #spherical is actual sphere. spheremap is equirectangular image 
    x=imgWidth * (1. - (theta * KPi2Inverted)) - 0.5     #considering it as a unit sphere
    y=(imgHeight * phi) * KPiInverted - 0.5
    x=np.round(x)
    y=np.round(y)
    return x.reshape(-1).astype(int),y.reshape(-1).astype(int)

# To load image
def load_image_info(image_path, imgshape=True):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Image shape info
    if imgshape==True:
        imgHeight,imgWidth = img.shape[:2]
        return img, imgHeight, imgWidth
    else:
        return img

####################################################################################

def matches_between_two_images(image0_path, image1_path, keypointCoords0, keypointCoords1, matches0):
    # Extracting image info
    img0, imgHeight0, imgWidth0 = load_image_info(image0_path, imgshape=True)
    img1, imgHeight1, imgWidth1 = load_image_info(image1_path, imgshape=True)

    # image 0 
    if keypointCoords0.shape[1] == 2:
        phi_kpts, theta_kpts = torch.split(keypointCoords0, 1, dim=1)
        x0,y0 = Spherical_to_Pixel_Coord(phi_kpts.numpy(), theta_kpts.numpy(), imgWidth0, imgHeight0)
        
        # image 1
        phi_kpts1, theta_kpts1 = torch.split(keypointCoords1, 1, dim=1)
        x1,y1 = Spherical_to_Pixel_Coord(phi_kpts1.numpy(), theta_kpts1.numpy(), imgWidth1, imgHeight1)

    if keypointCoords0.shape[1] == 3:
        theta_kpts, phi_kpts = hp.vec2ang(keypointCoords0)
        x0,y0 = Spherical_to_Pixel_Coord(theta_kpts, phi_kpts, imgWidth0, imgHeight0)

        # image 1
        theta_kpts1, phi_kpts1 = hp.vec2ang(keypointCoords1)
        x1,y1 = Spherical_to_Pixel_Coord(theta_kpts1, phi_kpts1, imgWidth1, imgHeight1)

    mask = matches0.ge(0)
    kpts_with_corr_img1 = torch.masked_select(torch.arange(matches0.shape[0]), mask)
    kpts_with_corr_img2 = torch.masked_select(matches0, mask)
    corresponding_kpts_index = torch.stack((kpts_with_corr_img1, kpts_with_corr_img2), 1)
    print(corresponding_kpts_index)

    v_concat_img = cv2.vconcat([img0, img1])
    for i in range(corresponding_kpts_index.shape[0]):
        ele = corresponding_kpts_index[i]
        v_concat_img = cv2.line(v_concat_img, (x0[ele[0].int().item()], y0[ele[0].int().item()]), (x1[ele[1].int().item()], imgHeight0+y1[ele[1].int().item()]), color=(255,0,0),thickness=2)

    return v_concat_img


# Spherical coordinates to Pixel coordinates for given NSIDE     
def Spherical_to_Pixel_Coord_for_given_nside(nside, imgWidth, imgHeight):  
    NPIX = hp.nside2npix(nside)
    ipix = np.arange(NPIX)
    theta, phi = hp.pix2ang(nside, ipix, nest=True)
    x1,y1 = Spherical_to_Pixel_Coord(theta, phi, imgWidth, imgHeight)
    return x1,y1


def keypoints_on_healpix_map_over_Single_image(image_path, nside, phi_kpts, theta_kpts):
    img, imgHeight, imgWidth = load_image_info(image_path)
    x1,y1 = Spherical_to_Pixel_Coord_for_given_nside(nside, imgWidth, imgHeight)
    keypoints_facets = set(hp.ang2pix(nside, phi_kpts, theta_kpts, nest=True).reshape(-1))

    for i in range(x1.shape[0]):
        if i in keypoints_facets:
            img = cv2.circle(img, (x1[i],y1[i]), radius=10, color=(255, 0, 0), thickness=-1)
        else:
            img = cv2.circle(img, (x1[i],y1[i]), radius=5, color=(0, 255, 0), thickness=1)
    return img