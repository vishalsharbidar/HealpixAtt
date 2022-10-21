
import numpy as np
import torch
from torch.utils.data import Dataset
import healpy as hp


#####################################################################################
def find_unique_ele(a):
    '''
    input 
        a: 1D-Tensor
        
    output
        unq_a: tensor of unique element in a, 
        idx: index of unique element
    '''
    seen = set()
    idx = []
    unq_a = []
    for ix, ele in enumerate(a):
        if ele.item() not in seen:
            seen.add(ele.item())
            unq_a.append(ele.item())
            idx.append(ix)
    return torch.tensor(unq_a), torch.tensor(idx), seen

def SphericalToCartesian(phi,theta, imgWidth, imgHeight):
    KPi2Inverted= 1/(2*np.pi)
    KPiInverted = 1/(np.pi)                          #spherical is actual sphere. spheremap is equirectangular image 
    x=imgWidth * (1. - (theta * KPi2Inverted)) - .5     #considering it as a unit sphere
    y=(imgHeight * phi) * KPiInverted - .5
    x=np.round(x)
    y=np.round(y)
    x=x.astype(int)
    y=y.astype(int)
    return x,y

def CartesianToSpherical(x,y,imgWidth,imgHeight):
    # Compute spherical theta coordinate
    Pi2= 2*np.pi
    Pi= np.pi
    theta = (1. - (x + .5) / imgWidth) * Pi2
    # Now theta is in [0, 2Pi]

    # Compute spherical phi coordinate
    phi = ((y + .5) * Pi) / imgHeight
    #Now phi is in [0, Pi]
    return (phi, theta)


def sphereMapCoordsToUnitCartesian( x, y, imgWidth, imgHeight):
    (phi, theta) = CartesianToSpherical(x, y, imgWidth, imgHeight)
    return list(sphericalToCartesian(phi, theta, 1))

##################### Spherical KNN ############################################################################################

# This function converts Spherical coordinates to Cartesian.........................................####
def sphericalToCartesian(phi, theta, radius):
    x = radius*torch.cos(theta)*torch.sin(phi) 
    y = radius*torch.sin(theta)*torch.sin(phi) 
    z = radius*torch.cos(phi)
    xyz = torch.stack((x,y,z), dim=1)
    return xyz

def calNpix(NSIDE):
    NPIX = hp.nside2npix(NSIDE)
    ipix = np.arange(NPIX)
    return ipix, NPIX

############################## End of Spherical KNN ############################################################################

############################## Start of function Profiling #####################################################################

import cProfile, pstats, io



def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

############################## End of function Profiling #######################################################################

############################## Creation of ground truth matrix for sinkhorn loss ###############################################
def corr2mat(data):    
    mat = -1*torch.ones((len(data['unitCartesian1'][0]),len(data['unitCartesian2'][0])), dtype=torch.int32)
    corr = data['correspondences'][0].detach().cpu().numpy()
    #print(mat.shape)
    for i in range(len(corr)):
        if corr[i]!=-1:
            mat[i][corr[i]]=1
    return mat

def smallmat(mat, idx1, idx2, device):
    bottleneck_idx1 = idx1[-2]
    bottleneck_idx2 = idx2[-2]
    new_mat = mat[bottleneck_idx1,:]
    new_mat = new_mat[:,bottleneck_idx2]
    new_mat = new_mat.squeeze(0).float().to(device)
    return new_mat

def gt_mat(data, idx1, idx2, device):
    matx = corr2mat(data)
    new = smallmat(matx, idx1, idx2, device)
    return new

################################################### End  #####################################################################

