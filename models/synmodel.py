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
from skimage.measure import compare_ssim
class SynANet(AdaptorNet):
    def __init__(self, opt):
         super(SynANet,self).__init__(opt)      
    def def_TNet(self):
        """Define Task Net for synthesis, segmentation e.t.c.
        """
        self.TNet = UNet(1,[64,64,64,64],1,isn=False)        
    def def_loss(self):
        self.TLoss = nn.MSELoss()
        self.AELoss = nn.MSELoss()
        self.ALoss = nn.MSELoss()
        self.LGanLoss = nn.BCEWithLogitsLoss()  
    def cal_metric(self, pred, label):
        """Calculate quantitative metric: 
        Args: 
            pred: [batch, channel, img_row, img_col]
            label:[batch, img_row, img_col]
        Return:
            MSE: mean squared error
            SSIM: structural similarity
        """  
        if torch.is_tensor(pred):
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
        metric = []
        batch_size = pred.shape[0]
        for b_id in range(batch_size):
            _metric = np.mean((pred[b_id] - label[b_id])**2)
            _metric = compare_ssim(pred[b_id], label[b_id])            
            metric.append(_metric)
        return metric        
        
    def test(self):
        self.TNet.eval()
        self.ANet.eval()
        with torch.no_grad():            
            pred = self.ANet(self.image, self.TNet, side_out=True)
            pred, adapt_img = pred[-1], pred[0]
            pred_na = self.TNet(self.image)
        if self.opt.saveimage:
            batch_size = self.image.shape[0]       
            for b_ix in range(batch_size):
                ids = os.path.split(self.filename[b_ix])[-1].split('.')[0]
                self.plot(tonp(adapt_img[b_ix]), ids + '_adapt.tif')
                self.plot(tonp(pred[b_ix]), ids + '_preda.tif')
                self.plot(tonp(pred_na[b_ix]), ids + '_predna.tif')
                self.plot(tonp(self.image[b_ix]), ids + '_image.tif')
                self.plot(tonp(self.label[b_ix]), ids + '_label.tif')
        metric = [[0]*batch_size,[0]*batch_size] 
        if self.opt.__dict__.get('cal_metric',True):
            metric[0] = self.cal_metric(pred.squeeze(1), self.label)
            metric[1] = self.cal_metric(pred_na.squeeze(1), self.label)
        return metric