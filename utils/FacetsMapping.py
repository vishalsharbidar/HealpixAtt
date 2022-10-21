import torch
import healpy as hp
import torch.nn as nn
import numpy as np
import time
from utils.Utils import sphericalToCartesian, calNpix
from sklearn.neighbors import KDTree


class HealpyFacetsMappingDict(nn.Module):
    def __init__(self):
        super().__init__()

    def Healpix_kdtree(self, NSIDE, NPIX):
        #Getting the values of phi and theta for all the pixels
        phi, theta = hp.pix2ang(nside=NSIDE, ipix=np.arange(NPIX), nest=True)
        # healpix facets cartesian coordinates
        healpix_xyz = sphericalToCartesian(torch.tensor(phi), torch.tensor(theta), 1)
        # Finding nearest neighbours
        healpix_kdtree = KDTree(healpix_xyz, leaf_size=30, metric='euclidean') 
        return healpix_kdtree

    def Healpix_pix2vec(self, NSIDE, NPIX, device):
        x, y, z = hp.pix2vec(nside=NSIDE, ipix=torch.arange(NPIX), nest=True)
        pix2vec = torch.stack([x, y, z], axis=1)
        return pix2vec.to(device)

    def forward(self, nsides, device):
        val2idx = {}
        idx2val = {}
        npix = {}
        ipix = {}
        kdtree = {}
        nested = {}
        pix2vec = {}
        for nside in nsides:
            npix_ = hp.nside2npix(nside)
            npix[nside]  = npix_
            ipix_ = torch.arange(npix_) 
            ipix[nside] = ipix_
            nested[nside] = hp.ring2nest(nside, ipix_)
            map_dict_all_facet = {f:i for i,f in enumerate(nested[nside])}
            val2idx[nside] = map_dict_all_facet
            map_dict_all_facet_idx2val = {i for i,f in enumerate(nested[nside])}            
            idx2val[nside] = map_dict_all_facet_idx2val
            kdtree[nside] = self.Healpix_kdtree(nside, npix_)
            pix2vec[nside] = self.Healpix_pix2vec(nside, npix_, device)
        return {#'val2idx':val2idx, 
                'idx2val':idx2val, 
                'npix':npix, 
                'ipix':ipix,
                #'nested_ipix' : nested, 
                'kdtree':kdtree, 
                'pix2vec':pix2vec}

if __name__ == '__main__':
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    func = HealpyFacetsMappingDict().to(device)
    nested_pix = [16, 8, 4, 2]
    s = time.time()
    a, d = func(nested_pix)
    e = time.time()
    print(e-s)
    #print(len(a[0]), len(a[1]), len(d))
    print(a.keys())


###############################################################################################################

 
    
