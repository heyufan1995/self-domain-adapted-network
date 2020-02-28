import os, glob
import torch
import numpy as np 
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import copy
import warnings
import logging
from scipy.io import loadmat
from scipy import misc
from copy import deepcopy
import pdb
logger = logging.getLogger('global')
warnings.filterwarnings("ignore")
from utils.util import generate_mask
import tifffile as tiff
import nibabel
class ABCDataset(Dataset):
    """Abstract Basic class for the dataset
    """
    def __init__(self,  transform=None, normalize_fn=None):
        # transformation
        self.transform = transform
        self.normalize_fn = normalize_fn
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self,idx):
        '''
        The format of the returned data is fixed
        '''
        # data: [img_rows, img_cols]
        # label: [img_rows, img_cols] 
        # filename: string, path to the image
        sample = self._get_data(idx)
        # augmentation
        if self.transform is not None:
            sample = self.transform(sample)
        # normalization
        if self.normalize_fn is not None:
            sample = self.normalize_fn(sample)
        # To tensor
        sample['data'] = torch.from_numpy(sample['data'].astype(np.float32)).unsqueeze(0)
        return sample     
    def _get_data(self,idx):
        raise NotImplementedError

class PairDataset(ABCDataset):
    '''Paired image dataset for synthesis '''
    def __init__(self,filepath=None, labelpath=None,\
                 file_ext=None, lab_ext=None, transform=None, normalize_fn=None):
        super(PairDataset,self).__init__(transform, normalize_fn)
        # use data and label folder path and extension
        self.filepath = filepath
        self.labelpath = labelpath
        self.file_ext = file_ext
        self.lab_ext = lab_ext
        # init from folder or txt 
        if filepath is not None:
            if os.path.isfile(filepath):
                with open(filepath) as f:
                    self.datalist = f.read().splitlines() 
            elif type(filepath) == list:
                self.datalist = filepath   
            elif type(filepath) == str:            
                self.datalist = sorted(list(Path(self.filepath).glob('*.'+self.file_ext)))
            if os.path.isfile(labelpath):
                with open(labelpath) as f:
                    self.labellist = f.read().splitlines() 
            elif type(labelpath) == list: 
                self.labellist = labelpath 
            elif type(labelpath) == str:                   
                self.labellist = sorted(list(Path(self.labelpath).glob('*.'+self.lab_ext)))
    def __len__(self):
        return len(self.datalist)
    def _get_data(self,idx):
        # implement _get_data method
        # the label comes from matlab, remenber -1 in boundary points
        if self.file_ext == 'tif':
            data = tiff.imread(str(self.datalist[idx])).astype(np.float32)
        elif self.file_ext == 'png':
            data = misc.imread(str(self.datalist[idx]),mode = 'L')
            data = (data/255).astype(np.float32)
        elif self.file_ext == 'nii.gz':
            data = nibabel.load(str(self.datalist[idx])).get_data().T
        label = data # default is reconstruction
        if len(self.labellist) > 0:
            if self.lab_ext == 'tif':
                label = tiff.imread(str(self.labellist[idx])).astype(np.float32)
            elif self.lab_ext == 'png':
                label = misc.imread(str(self.labellist[idx]),mode = 'L')
                label = (label/255).astype(np.float32)
            elif self.lab_ext == 'nii.gz':
                label = nibabel.load(str(self.labellist[idx])).get_data().T
        data = (data - np.mean(data))/np.std(data)
        label = (label - np.mean(label))/np.std(label)
        sample = {'data':data, 'label':label, 'filename':str(self.datalist[idx])}
        return sample

class SegDataset(PairDataset):
    def __init__(self,filepath=None, labelpath=None,\
                 file_ext=None, lab_ext=None, transform=None, normalize_fn=None):
        super(SegDataset,self).__init__(filepath, labelpath,\
                 file_ext, lab_ext, transform, normalize_fn)
    def _get_data(self,idx):
        data = misc.imread(str(self.datalist[idx]),mode = 'L')
        data = (data/255).astype(np.float32)
        if len(self.labellist) > 0:
            # allow none label
            with open(str(self.labellist[idx]),'r') as f:
                dicts = json.loads(f.read())
            if 'lesion' in dicts.keys():
                mask = np.array(dicts['lesion'])
                mask[mask>1] = 1
            else:
                mask = np.zeros(data.shape)
            # add dtype=float to make sure None is converted to NaN
            bds = np.array(dicts['bds'], dtype=np.float) - 1 
            mask = generate_mask(bd_pts=bds,lesion=mask)
        else:
            mask = np.zeros(data.shape)
            bds = np.zeros((9,data.shape[1]))
        # limit bds within image range
        bds[bds<0] = 0
        bds[bds>data.shape[0]-1] = data.shape[0] - 1
        # normalize input to have zero mean unit variance
        data = (data - np.mean(data))/np.std(data)
        sample = {'data':data, 'label':mask, 'filename':str(self.datalist[idx])}
        return sample

def split_data(dataset,split,switch=False):
    ''' split training data into training/validation
        Args: 
            split[0] - split[1] val
            split[1] - split[2] train
            switch: switch returned dataset
    '''
    traindataset = deepcopy(dataset)
    valdataset = deepcopy(dataset)
    if len(split)>2:
        idx = np.arange(split[0],split[-1])
    else:
        idx = np.arange(len(dataset))
    validx = np.arange(int(split[0]),int(split[1]),dtype=np.uint8)
    traidx = np.array(list(set(idx)-set(validx)),dtype=np.uint8)
    traindataset.datalist = [dataset.datalist[i] for i in traidx]
    traindataset.labellist = [dataset.labellist[i] for i in traidx]
    valdataset.datalist = [dataset.datalist[i] for i in validx]
    valdataset.labellist = [dataset.labellist[i] for i in validx]
    if switch: # switch train/val to return the correct test set
        return valdataset, traindataset
    else:
        return traindataset, valdataset      




    
