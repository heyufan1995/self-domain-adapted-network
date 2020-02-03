import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb
from models.backends import UNet, ConvBlock, DenseBlock
from utils.util import ncc, grams
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import logging
logger = logging.getLogger('global')
from models.basemodel import tonp, AdaptorNet
 
class SegANet(AdaptorNet):
    def __init__(self, opt):
         super(SegANet,self).__init__(opt)
    def def_TNet(self):
        """Define Task Net for synthesis, segmentation e.t.c.
        """
        self.TNet = UNet(1,[64,64,64,64],11,isn=False)        
    def def_loss(self):
        self.TLoss = nn.CrossEntropyLoss()
        self.AELoss = nn.MSELoss()
        self.ALoss = nn.MSELoss()   
        self.LGanLoss = nn.BCEWithLogitsLoss()
    def cal_metric(self, pred, label, ignore=[0,9,10]):
        """Calculate quantitative metric: Dice coefficient
        Args: 
            pred: [batch, channel, img_row, img_col]
            label: [batch, img_row, img_col]
            ignore: ignore labels
        Return:
            dice: list of dice coefficients, [batch]
        """  
        if torch.is_tensor(pred):
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
        dice = []
        batch_size = pred.shape[0]
        for b_id in range(batch_size):
            C = pred[b_id].shape[0]
            _dice = np.zeros(C)
            for i in range(C):
                pl = pred[b_id,i,:,:]
                ll = label[b_id,:,:] == i
                sump = np.nansum(pl)
                sumt = np.nansum(ll)
                inter = np.nansum(pl*ll)
                if sumt == 0:
                    _dice[i] = np.nan
                else:
                    _dice[i] = 2*inter/(sump+sumt)  
            _dice = np.delete(_dice,ignore)
            dice.append(_dice)    
        return dice
    def test(self):
        """Test using ANet and TNet
        """
        self.TNet.eval()
        self.ANet.eval()
        with torch.no_grad():
            pred = self.ANet(self.image, self.TNet, side_out=True)
            pred, adapt_img = pred[-1], pred[0]
            pred = F.softmax(self.ANet(self.image, self.TNet), dim=1)
            pred_na = F.softmax(self.TNet(self.image), dim=1)
        batch_size = self.image.shape[0]   
        if self.opt.saveimage:
            for b_ix in range(batch_size):
                ids = os.path.split(self.filename[b_ix])[-1].split('.')[0]
                self.plot(tonp(adapt_img[b_ix]), ids + '_adapt.png')
                self.plot(tonp(pred[b_ix]), ids + '_preda.png')
                self.plot(tonp(pred_na[b_ix]), ids + '_predna.png')
                self.plot(tonp(self.image[b_ix]), ids + '_image.png')
        metric = [[],[]]            
        if self.opt.__dict__.get('cal_metric',True):
            metric[0] = self.cal_metric(pred, self.label.squeeze(1))
            metric[1] = self.cal_metric(pred_na, self.label.squeeze(1))
        return metric
