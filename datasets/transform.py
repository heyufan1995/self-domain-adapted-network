import numpy as np 
import cv2
import pdb
import logging
logger = logging.getLogger('global')
from matplotlib import pyplot as plt
from scipy import interpolate
class OCTTransform(object):
    def __init__(self, scales=None, crop=None, flip=False , valcrop=None, addnan=None):
        # scales are the min and max height
        self.scales=scales                
        self.flip = flip
        self.crop = crop
        # used in validation. crop the image due to memory issue.
        self.valcrop = valcrop
        self.addnan = addnan
    def __call__(self, sample):
        """
        Args: 
        data [img_rows, img_cols]
        bds [bdc, img_cols]
        mask [img_rows, img_cols]
        """
        data, bds, mask = sample['data'], sample['bds'], sample['mask']
        h,w = data.shape
        if self.addnan:
            # the DME dataset does not have ELM layer
            # add it to the DME data 
            # make sure it's DME
            assert bds.shape[0] == 8
            bds = np.vstack([bds[:-3,:],np.zeros(bds.shape[-1])*np.nan,bds[-3:,:]])
            mask[mask>5] += 1
        if self.flip:
            if np.random.random() < 0.5:
                # copy() needed for negative stride problem in pytorch from_numpy
                # may be solved by future pytorch version
                data = np.flip(data,-1).copy()
                bds = np.flip(bds,-1).copy()
                mask = np.flip(mask,-1).copy()
        
        h_n = h
        if self.scales:
            # scale [min_height, max_heigh, mode]
            # mode 0: retina scaling inside image and random crop, used for DME
            # mode 1: the choroid is not crropped.
            # mode 2: rescale the image to a fixed size, default 512x512, used
            #         in no flattening setting 
            if len(self.scales) == 2:
                self.scales.append(0)
            if self.scales[-1] == 2:
                sizes = data.shape
                data = cv2.resize(data, tuple(self.scales[:-1]), interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, tuple(self.scales[:-1]), interpolation=cv2.INTER_NEAREST)
                if sizes[0] != self.scales[0]:
                    bds = bds*self.scales[0]/sizes[0]
                if sizes[1] != self.scales[1]:
                    f = interpolate.interp1d(np.arange(sizes[1]), bds, axis=-1)
                    xnew = np.linspace(0, sizes[1]-1, self.scales[1])
                    bds = f(xnew)                    
            else:
                if np.random.random() < 0.5:
                    # make sure the retina is inside the scale
                    top = np.nanmin(bds[0,:])
                    bot = h
                    if self.crop is None:
                        self.crop = [h,w]
                    margin = 3                
                    self.scales[1]= max(min((self.crop[0] - 2*margin)/(bot - top) * h,
                                             self.scales[1]), self.scales[0]) 
                    h_n = np.random.randint(self.scales[0], int(self.scales[1]) + 1)
                    scale = h_n/h
                    data = cv2.resize(data, (w, h_n), interpolation=cv2.INTER_CUBIC)
                    mask = cv2.resize(mask, (w,h_n), interpolation=cv2.INTER_NEAREST)
                    bds = bds*scale

        # crop the image to the original size or to the crop size
        if self.crop:
            if self.scales and self.scales[-1] == 1:
                clims_w = w//2 
            else:
                b_sum = bds[0]            
                clims_left = np.argmax(~np.isnan(b_sum)) + self.crop[1]//2
                clims_right = w - np.argmax(~np.isnan(b_sum[::-1])) - self.crop[1]//2
                # make sure the crop center is not on a nan boundary
                # due to the nan value of DME manual delineation
                idx = np.arange(len(b_sum))
                idx = np.logical_and(idx>=clims_left, idx<=clims_right)
                idx = np.logical_and(idx,~np.isnan(b_sum))
                idx = np.where(idx>0)[0]
                clims_w =  np.random.choice(idx)
            clims_h = h_n - self.crop[0]//2
            data, bds, mask = crop([data, bds, mask],[clims_h,clims_w],self.crop)
        bds[bds<0] = 0
        bds[bds>data.shape[0]-1] = data.shape[0] -1            
        sample['data'], sample['bds'], sample['mask'] = data, bds, mask
        if self.valcrop:
            # Crop the input image to a smaller size for less memory usage
            # Need to make sure the valcrop paramsters will fit for the retina
            #clims_w = (clims_left + clims_right)//2
            #clims_w = idx[np.argmin(abs(clims_w - idx))]
            #clims_h = (bds[0,clims_w] + bds[-1,clims_w])//2
            clims_h = h//2 - self.valcrop[0]//2 
            clims_w = w//2 - self.valcrop[1]//2
            sample['nocrop'] = {'data': data,'bds':bds,'mask':mask}
            data = data[clims_h:clims_h+self.valcrop[0], clims_w:clims_w+self.valcrop[1]]
            mask = mask[clims_h:clims_h+self.valcrop[0], clims_w:clims_w+self.valcrop[1]]
            bds = bds[:,clims_w:clims_w+self.valcrop[1]] - clims_w
            #data, bds, mask = crop([data, bds, mask],[clims_h,clims_w],self.valcrop)
            sample['data'], sample['bds'], sample['mask'] = data, bds, mask

        return sample
class OCTNormalizer():
    ''' Normalize the input 2D B-scan with zero-mean unit variance'''
    def __init__(self,paras=[0,1], vol=False):
        self.paras = paras
        self.vol = vol

    def __call__(self,sample):
        if self.vol:
            # [B,1,H,W]
            data = sample
            if len(self.paras)>0:
                data = data - np.mean(data,(-1,-2),keepdims=True) \
                        + self.paras[0]
            if len(self.paras)>1:
                data = data/np.std(data,(-1,-2),keepdims=True)*self.paras[1]            
            return data
        data = sample['data']
        # mask out bright artifacts
        mask_data = data[data<0.99]
        if len(self.paras)>0:
            data = data - np.mean(mask_data) + self.paras[0]
        if len(self.paras)>1:
            data = data/np.std(mask_data)*self.paras[1]
        sample['data'] =data
        return sample


def crop(sample,center,sizes):
    """
    Args: 
    data [img_rows, img_cols]
    bds [bdc, img_cols]
    mask [img_rows, img_cols]
    sizes: crop patch size
    center: patch center
    """
    data,bds,mask = sample
    s = sizes
    r,c = center
    img_rows,img_cols = data.shape
    imgc = np.zeros(s)
    maskc = np.zeros(s)*np.nan
    bdsc = np.zeros((bds.shape[0],s[1]))*np.nan
    # build row index corespondence between patch and image
    rr_idx = (np.arange(s[0])).astype(int)
    r_idx = (rr_idx -s[0]//2 + r).astype(int)
    r_mask = (r_idx>=0) & (r_idx< img_rows)
    # build col index corespondence between patch and image
    cc_idx = (np.arange(s[1])).astype(int)
    c_idx = (cc_idx -s[1]//2 + c)
    c_mask = (c_idx>=0) & (c_idx< img_cols)

    X,Y = np.meshgrid(r_idx[r_mask],c_idx[c_mask])
    XX,YY = np.meshgrid(rr_idx[r_mask],cc_idx[c_mask])

    imgc[XX.flatten(),YY.flatten()] = data[X.flatten(),Y.flatten()]
    maskc[XX.flatten(),YY.flatten()] = mask[X.flatten(),Y.flatten()]
    bdsc[:,cc_idx[c_mask]] = bds[:,c_idx[c_mask]] - int(r) + s[0]//2
    return [imgc,bdsc,maskc]





        
        


