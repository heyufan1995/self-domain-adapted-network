import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb
from models.backends import UNet, ConvBlock, DenseBlock
from utils.util import ncc
from external.ssim import pytorch_ssim
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import logging
logger = logging.getLogger('global')
from models.basemodel import tonp, AdaptorNet

class ANet(nn.Module):
    def __init__(self, ndf=64, dilations=[1,1,1,1]):
        super(ANet, self).__init__()
        self.dense = DenseBlock(1,ndf,4,dilations=dilations)
        self.conv3 = nn.Conv2d(4*ndf+1,1,kernel_size=1)
    def forward(self, x):
        x = self.dense(x)
        x = self.conv3(x)
        return x
class bottleneck(nn.Module):
    def __init__(self, channel, poolsize=[32,32]):
        super(bottleneck,self).__init__()
        dims = channel*poolsize[0]*poolsize[1]
        self.poolsize = poolsize
        # self.pool = nn.AdaptiveAvgPool2d(poolsize)
        self.pool = nn.Upsample(size=poolsize, mode='bilinear')
        self.ae = nn.Sequential(
            nn.Linear(dims, dims//2),
            nn.PReLU(),
            nn.Linear(dims//2, dims))
    def forward(self,x):
        ori_size = x.shape
        x = self.pool(x)
        x = self.ae(x.view(ori_size[0],-1))
        x = x.view(ori_size[0], ori_size[1], self.poolsize[0], self.poolsize[1])
        x = F.upsample(x, size=ori_size[-2:], mode='bilinear')
        return x
class SynANet(AdaptorNet):
    def __init__(self, opt):
         super(SynANet,self).__init__(opt)
    def def_ANet(self):
        self.ANet = ANet(dilations=[1,1,1,1])  
        # self.ANet = nn.Sequential(
        #     nn.Conv2d(1,64,1),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(64),
        #     nn.Conv2d(64,128,1),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(128),
        #     nn.Conv2d(128,64,1),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(64),
        #     nn.Conv2d(64,1,1)
        # )   
    def def_TNet(self):
        """Define Task Net for synthesis, segmentation e.t.c.
        """
        self.TNet = UNet(1,[64,64,64,64],1,isn=True)
    def opt_AENet(self):
        """Optimize Auto-Encoder seperately
        """
        self.set_requires_grad(self.TNet,False)
        self.set_requires_grad(self.AENet,True)
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.train()
        side_out = list(map(self.TNet(self.image,side_out=True).__getitem__, self.AENetMatch))     
        ae_out = [self.AENet[_](side_out[_]) for _ in range(len(self.AENet))]
        self.optimizer_AENet.zero_grad()
        loss = 0
        weights = self.opt.__dict__.get('weights', [1]*len(ae_out))
        for _ in range(len(ae_out)):
            loss += weights[_]*self.AELoss(ae_out[_], side_out[_]) 
        loss.backward()
        self.optimizer_AENet.step()
        return loss.data.item()        
    def def_AENet(self):
        """Define Auto-Encoder for training on source images
        """
        # exp3
        # self.AENet = [UNet(1,[32,16,8],1,isn=True,skip=False),
        #               UNet(64,[32,16,8],64,isn=True,skip=False),
        #               UNet(64,[32,16,8],64,isn=True,skip=False),
        #               UNet(1,[32,16,8],1,isn=True,skip=False)]        
        # exp 6
        # self.AENet = [UNet(1,[16,8],1,isn=True,skip=False,
        #               bottleneck=bottleneck(channel=8)),
        #               UNet(64,[32,16,8],64,isn=True,skip=False),
        #               UNet(64,[32,16,8],64,isn=True,skip=False),
        #               UNet(1,[16,8],1,isn=True,skip=False,
        #               bottleneck=bottleneck(channel=8))]
        # exp7
        # self.AENet = [UNet(1,[32,16,8],1,isn=True,skip=False),
        #               UNet(64,[32,16,8],64,isn=True,skip=False),
        #               UNet(64,[32,16,8],64,isn=True,skip=False),
        #               UNet(1,[32,16,8],1,isn=True,skip=False)]     
        # exp 8
        self.AENet = [UNet(64,[32,16,8],64,isn=True,skip=False),
                      UNet(64,[32,16,8],64,isn=True,skip=False),
                      UNet(64,[32,16,8],64,isn=True,skip=False),
                      UNet(64,[32,16,8],64,isn=True,skip=False)]          
        self.AENetMatch = [-2,-3,-4,-5]
        assert len(self.AENet) == len(self.AENetMatch)    
    def def_loss(self):
        self.TLoss = nn.MSELoss()
        self.AELoss = nn.MSELoss()
        self.ALoss = nn.MSELoss()
    def test(self):
        self.TNet.eval()
        self.ANet.eval()
        with torch.no_grad():
            adapt_img = self.ANet(self.image)
            pred = self.TNet(adapt_img)
            pred_na = self.TNet(self.image)
        if self.opt.saveimage:
            batch_size = self.image.shape[0]       
            for b_ix in range(batch_size):
                ids = os.path.split(self.filename[b_ix])[-1].split('.')[0]
                self.plot(tonp(pred[b_ix]), ids + '_preda.png')
                self.plot(tonp(pred_na[b_ix]), ids + '_predna.png')
                self.plot(tonp(self.image[b_ix]), ids + '_image.png')
                self.plot(tonp(adapt_img[b_ix]), ids + '_adapt.png') 
                if self.opt.__dict__.get('hist', False):
                    self.plot_hist([tonp(self.image[b_ix]),tonp(adapt_img[b_ix])],
                                    ids + '_hist2d.png')
                    self.plot_hist([tonp(self.image[b_ix])], ids + '_hist_ori.png')  
                    self.plot_hist([tonp(adapt_img[b_ix])], ids + '_hist_adp.png')   
        metric = [[0]*batch_size,[0]*batch_size] 
        if self.opt.__dict__.get('cal_metric',False):
            metric[0] = self.cal_metric(pred, self.label.squeeze(1))
            metric[1] = self.cal_metric(pred_na, self.label.squeeze(1))
        return metric