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
def tonp(x):
    return x.squeeze().data.cpu().numpy()
class AdaptorNet(nn.Module):
    def __init__(self, opt):
        super(AdaptorNet,self).__init__()
        self.opt = opt
        self.def_TNet()
        self.def_AENet()
        self.def_ANet()
        self.def_loss()
        self.set_opt()
    def set_opt(self):
        # set TNet optimizers
        self.TNet.cuda()
        self.optimizer_TNet = torch.optim.Adam(self.TNet.parameters(), 
                                               lr=self.opt.tlr)
        # set AENet optimizers
        params = [] 
        for subnets in self.AENet:
            subnets.cuda()
            params.extend(list(subnets.parameters()))
        self.optimizer_AENet = torch.optim.Adam(params, self.opt.aelr)
        # set ANet optimizers
        self.ANet.cuda()
        self.optimizer_ANet = torch.optim.Adam(self.ANet.parameters(),
                                               self.opt.alr)
    def def_TNet(self):
        """Define Task Net for synthesis, segmentation e.t.c.
        """
        pass
    def def_AENet(self):
        """Define Auto-Encoder for training on source images
        """
        pass
    def def_ANet(self):
        """Define Adaptor Net for domain adapt
        """
        pass
    def def_loss(self):
        """Define training loss
        examples:
            self.TLoss = nn.MSELoss()
            self.AELoss = nn.MSELoss()
            self.ALoss = nn.MSELoss()
        """
        pass
    def set_requires_grad(self, nets, requires_grad=False, cuda=True):
        """Set requies_grad=Fasle and move to cuda
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                if cuda:
                    net.cuda()
                for param in net.parameters():
                    param.requires_grad = requires_grad    
    def set_input(self, data):
        """Unpack input data and perform necessary pre-processing steps.
        self.image: [batch, 1, img_rows, img_cols]
        self.label: [batch, img_rows, img_cols]
        self.filename: str
        """    
        self.image = torch.stack(tuple(data['data'])).cuda()
        self.label = torch.stack(tuple(data['label'])).cuda()
        self.filename = data['filename']    
    def opt_TNet(self):      
        """Optimize Task Net seperately
        """  
        self.set_requires_grad(self.TNet,True)
        self.TNet.train()
        pred = self.TNet.forward(self.image,side_out=False)
        self.optimizer_TNet.zero_grad()
        loss = self.TLoss(pred, self.label.long())
        loss.backward()
        self.optimizer_TNet.step()
        return loss.data.item()
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
    def opt_TAENet(self):
        """Optimize Auto-Encoder and Task Net jointly
        """
        pass
    def opt_ANet(self,epoch):
        """Optimize Adaptor
        """
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.eval()        
        self.set_requires_grad([self.TNet] + self.AENet, False) 
        self.set_requires_grad(self.ANet, True) 
        self.ANet.train()
        adapt_img = self.ANet(self.image)
        side_out = list(map(self.TNet(adapt_img,
                                      side_out=True).__getitem__, self.AENetMatch))     
        ae_out = [self.AENet[_](side_out[_]) for _ in range(len(self.AENet))]
        self.optimizer_ANet.zero_grad()
        loss = 0
        weights = self.opt.__dict__.get('weights', [1]*len(ae_out))        
        if self.opt.task == 'syn':
            for _ in range(len(ae_out)):
                loss += weights[_]*self.ALoss(ae_out[_], side_out[_]) 
        elif self.opt.task == 'seg':
            if epoch > 10:
                for _ in range(len(ae_out)):
                    loss += weights[_]*self.ALoss(ae_out[_], side_out[_]) 
            else:  
                loss += nn.MSELoss()(adapt_img, self.image)  
        loss.backward()
        self.optimizer_ANet.step()
        return loss.data.item()
    def plot(self,image,ids):
        """Plot and save image
        Args:
            image: [channel, img_rows, img_cols] for segmentation masks
                   [img_rows, img_cols] for synthesis
        """
        os.makedirs(os.path.join(self.opt.results_dir,'image'), exist_ok=True)
        fig, ax = plt.subplots(1, 1)
        if len(image.shape) > 2:
            ax.imshow(np.argmax(image,axis=0))
        else:
            ax.imshow(image,cmap='gray')
        fig.savefig(os.path.join(self.opt.results_dir,'image',ids),dpi=self.opt.dpi)
    def plot_hist(self,image,ids, xlim=[-1.5,3]):
        """Plot joint histogram
        Args:
            image: list of images
        """
        os.makedirs(os.path.join(self.opt.results_dir,'image'), exist_ok=True)
        fig, ax = plt.subplots(1, 1)
        if len(image) > 1:
            ax.hist2d(image[0].flatten(),image[1].flatten(), bins=50, normed=True,
                      cmax=1)
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
        else:
            ax.hist(image[0].flatten(), bins=50, density=True, facecolor='g')
            ax.set_xlim(xlim)
            ax.set_ylim([0,1])
        fig.tight_layout()
        fig.savefig(os.path.join(self.opt.results_dir,'image',ids),dpi=self.opt.dpi)
    def test(self):
        """Test using ANet and TNet
        """
        pass      
    def eval(self):
        """Evaluation on TNet and AENet
        """
        self.TNet.eval()
        with torch.no_grad():
            pred = self.TNet.forward(self.image)
            loss = self.Tloss(pred,self.label)
        if self.opt.saveimage:
            batch_size = self.image.shape[0]       
            for b_ix in range(batch_size):
                ids = os.path.split(self.filename[b_ix])[-1].split('.')[0]
                self.plot(pred[b_ix].squeeze().data.cpu().numpy(), ids + '_pred.png')
                self.plot(self.image[b_ix].squeeze().data.cpu().numpy(), ids + '_image.png')
                self.plot(self.label[b_ix].squeeze().data.cpu().numpy(), ids + '_label.png')
        return loss
    def save_nets(self, epoch):
        """Save network weights
        """
        os.makedirs(os.path.join(self.opt.results_dir, 'checkpoints'), exist_ok=True)
        save_path = os.path.join(self.opt.results_dir, 'checkpoints', 
                                 self.opt.trainer + '_checkpoint_e%d.pth' % (epoch + 1))
        save_dicts = {'epoch': epoch + 1}
        if self.opt.trainer == 'tnet':
            save_dicts['state_dict'] = self.TNet.cpu().state_dict()
            save_dicts['optimizer'] = self.optimizer_TNet.state_dict()
        else:       
            save_dicts['state_dict'] = [_.cpu().state_dict() for _ in self.AENet]
            save_dicts['optimizer'] = self.optimizer_AENet.state_dict()
        torch.save(save_dicts, save_path)        

    def load_nets(self, checkpoint, name='tnet'):
        """Load network weights
        """
        ckpt = torch.load(checkpoint)
        state_dict = ckpt['state_dict']
        optimizer = ckpt['optimizer']
        if name is 'tnet':        
            self.TNet.load_state_dict(state_dict)  
            self.optimizer_TNet.load_state_dict(optimizer)
            logger.info('TNet weights loaded')
        elif name is 'aenet':
            for _ in range(len(self.AENet)):
                self.AENet[_].load_state_dict(state_dict[_])
            self.optimizer_AENet.load_state_dict(optimizer)
            logger.info('AENet weights loaded')
        else:
            raise TypeError('Unknown model name')

