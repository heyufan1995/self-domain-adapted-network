import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb
from models.backends import UNet, ConvBlock, DenseBlock
from utils.util import ncc, l2_reg_ortho
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import logging
import tifffile as tiff
from functools import partial
logger = logging.getLogger('global')
def tonp(x):
    return x.squeeze().data.cpu().numpy()
def init_weights(x):
    torch.manual_seed(0)
    if type(x) == nn.Conv2d:
        nn.init.kaiming_normal_(x.weight.data)
        nn.init.zeros_(x.bias.data)         
def init_weights_eye(x, channel=64):
    # indentity init, only works for same input/output channel
    eye = nn.init.eye_(torch.empty(channel, channel)).unsqueeze(-1).unsqueeze(-1)
    init_bias = nn.init.zeros_(torch.empty(channel))            
    if type(x) == nn.Conv2d:
        x.weight.data = eye
        x.bias.data = init_bias   
class ANet(nn.Module):
    def __init__(self,channel=64, nums=8, adpt=False, adpNet=None, seq=None):
        """Adaptor Net
        Args:
            channel: int, input feature channel of the affine transform
            nums: int, number of affine transform matrix
            adpt: bool, use front adaptor
            adpNet: nn.Module, user defined front adaptor
            seq: list->int, index of the affine matrix which will be used 
        """
        super(ANet,self).__init__()
        self.conv = nn.ModuleList()
        self.channel = channel
        if seq is None:
            self.seq = np.arange(nums)  
        else:
            self.seq = seq
        self.nums = nums
        self.adpt = adpt
        self.seq = seq 
        # use pre-contrast manipulation 
        if adpt:
            if adpNet is None:
                self.adpNet = nn.Sequential(
                    nn.Conv2d(1,64,1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.InstanceNorm2d(64),
                    nn.Conv2d(64,64,1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.InstanceNorm2d(64),   
                    nn.Conv2d(64,1,1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.InstanceNorm2d(1)                    
                )
                self.adpNet.apply(init_weights)
            else:
                self.adpNet = adpNet
        # use feature affine transform
        for _ in range(nums):
            convs = nn.Conv2d(channel,channel,1)
            self.conv.append(convs)
        self.conv.apply(init_weights_eye)
    def reset(self):
        self.conv.apply(init_weights_eye)
        if self.adpt and self.adpNet is not None:
            self.adpNet.apply(init_weights)
        self.cuda()
    def forward(self, x, TNet, side_out=False):
        """
        Args: 
            TNet: nn.Module. The pretrained task network
            side_out: bool. If true, output every intermediate results
            seq: list->int or np array. Position of 1x1 convolution
        """
        if self.adpt:
            x = self.adpNet(x)
        xh = [x]      
        x = TNet.inblocks(x)
        ct = 0
        # apply 1x1 conv on input blocks
        seq = self.seq
        if ct in seq:
            x = self.conv[ct](x)
        ct += 1
        xh.append(x)
        for i in range(TNet.depth):
            x = TNet.downblocks[i](x)    
            # apply 1x1 conv on every downsample output except bottleneck       
            if ct in seq:
                x = self.conv[ct](x)
            ct += 1            
            xh.append(x)                    
        x = TNet.bottleneck(x)
        if ct in seq:
            x = self.conv[ct](x)
        ct += 1          
        xh.append(x)
        for i in range(TNet.depth):
            x = TNet.upblocks[TNet.depth-i-1](x,xh[TNet.depth-i])
            if ct in seq:
                x = self.conv[ct](x)
            ct += 1            
            xh.append(x)        
        x = TNet.outblock(x)
        xh.append(x)
        if side_out:
            return xh
        else:
            return x 
class BotNeck(nn.Module):
    def __init__(self, channel, f_sz):
        super(BotNeck,self).__init__()
        in_feat = channel * f_sz[0] * f_sz[1]
        self.fc = nn.Sequential(
            nn.Linear(in_feat, in_feat//8),
            nn.ReLU(),
            nn.Linear(in_feat//8, in_feat//8),
            nn.ReLU(),
            nn.Linear(in_feat//8, in_feat)
        )
    def forward(self,x):
        b, c, h, w =  x.shape
        x = self.fc(x.view(b,-1))
        x = x.view(b,c,h,w)
        return x
class AENet(nn.Module):
    """Flexible Auto Encoder
    The bottleneck in default is a convolution and the input size is flexible.W
    Args:
        channel: input feature channel
        midplane: if not defined, will decrease input channel by 2
        nums: number of maxpooling + 1
        bottleneck: nn.module, bottlenck of the UNet.
                    kwargs argument, default is a 1x1 conv
        isn: bool, use Instance Normalization.
             kwargs argument, default is False
        skip: bool, use long skip connection in UNet. 
              kwargs argument, default is True, but manually set to False here
        kwargs: other parameters that can be used by backends/UNet
    """
    def __init__(self, channel=128, midplane=[],\
                 nums=3, **kwargs):
        super(AENet,self).__init__()
        if not midplane:
            for _ in range(1,nums+1):
                midplane.append(channel//(2**_))
            botchannel = channel//2**nums
        else:
            botchannel = midplane[-1]
            nums = len(midplane)
        self.nums = nums
        kwargs.setdefault('skip', False)
        kwargs.setdefault('isn', True)
        self.unet = UNet(channel,midplane,channel,**kwargs)  
    def forward(self,x,side_out=False):
        side_outs = self.unet(x,side_out=True)      
        rec_out = side_outs[-1]
        if side_out:
            # return the bottleneck encoding for further regularization
            bot_out = side_outs[self.nums + 1]          
            return bot_out, rec_out            
        else:
            return rec_out
class AdaptorNet(nn.Module):
    """Base model
    """
    def __init__(self, opt):
        super(AdaptorNet,self).__init__()
        self.opt = opt
        self.def_TNet()
        self.def_AENet()
        self.def_ANet()
        self.def_LGan()
        self.def_loss()
        self.set_opt()
    def set_opt(self):
        """Move models to GPU and set optimizers.
        """
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
        # set LGan optimizers
        self.LGan.cuda()
        self.optimizer_LGan = torch.optim.Adam(self.LGan.parameters(),
                                               self.opt.llr)
    def def_TNet(self):
        """Define Task Net for synthesis, segmentation e.t.c.
        """
        pass
    def def_AENet(self):
        """Define Auto-Encoder for training on source images
           Network definition could be moved to config.json
        6 level AE list:
        level 0: input image(seq: None),
                 l0-down-feature(seq:0) + l0-up-feature(seq:7),
                 output image(seq: None)
        level 1: l1-down-feature(seq:1) + l1-up-feature(seq:6)
        level 2: l2-down-feature(seq:2) + l2-up-feature(seq:5)
        level 3: l3-down-feature(seq:3) + l3-up-feature(seq:4)
        """
        # only use the highest-level feature 
        if self.opt.pad_size or self.opt.scale_size:
            if self.opt.scale_size:
                sz = self.opt.scale_size        
            else:
                sz = self.opt.pad_size
        if self.opt.task == 'syn':out = 1 
        else:out = 11
        self.AENet = [AENet(channel=1,midplane=[32,16,8]),\
                      AENet(channel=128,midplane=[64,32,16]),\
                      AENet(channel=128,midplane=[64,32,16]),\
                      AENet(channel=128,midplane=[64,32,16]),\
                      AENet(channel=128,midplane=[64,32,16]),\
                      AENet(channel=out,midplane=[32,16,8])]              
        # the matching index of TNet features          
        self.AENetMatch = [[0],[1,-2],[2,-3],[3,-4],[4,-5],[-1]] 
    def def_ANet(self):
        """Define Adaptor Net for domain adapt
        """
        self.ANet = ANet(adpt=not self.opt.no_adpt,seq=self.opt.seq)
    def def_LGan(self):
        """Define latent space discriminator
        """
        inplane = 128
        self.LGan = nn.Sequential(
            nn.Linear(inplane,inplane//2),
            nn.LeakyReLU(),
            nn.Linear(inplane//2,inplane//4),
            nn.LeakyReLU(),
            nn.Linear(inplane//4,inplane//8),
            nn.LeakyReLU(),
            nn.Linear(inplane//8,1),     
        )    
    def def_loss(self):
        """Define training loss
        examples:
            self.TLoss = nn.MSELoss()
            self.AELoss = nn.MSELoss()
            self.ALoss = nn.MSELoss()
            self.LGanLoss = nn.BCEWithLogitsLoss()
        """
        pass
    def set_requires_grad(self, nets, requires_grad=False, cuda=True):
        """Set requies_grad=False and move to cuda
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                if cuda:
                    net.cuda()
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def set_data_pre(self,**kwargs):
        """Preprocess the input image and labels
           Supports cropping/padding/scaling
           Images from different dataset needs to have the same
           physical resolution and pre-registered to some common
           space.
        """    
        if 'image' not in self.__dict__: return 
        if self.opt.pad_size:
            h_n, w_n = self.opt.pad_size
            h,w = self.image.shape[-2:]
            # crop in the middel
            if h > h_n:
                self.image = self.image[:,:,(h-h_n//2):(h-h_n//2)+h_n, :]
                self.label = self.label[:,(h-h_n//2):(h-h_n//2)+h_n, :]
            if w > w_n:
                self.image = self.image[:,:,:,(w-w_n//2):(w-w_n//2)+w_n]
                self.label = self.label[:,:,(w-w_n//2):(w-w_n//2)+w_n]
            h,w = self.image.shape[-2:]
            # padding 
            pd = tuple(map(int,[(w_n-w)//2, w_n-w-(w_n-w)//2, (h_n-h)//2, h_n-h-(h_n-h)//2]))
            self.image = nn.ReplicationPad2d(pd)(self.image)
            self.label = nn.ReplicationPad2d(pd)(self.label.unsqueeze(1)).squeeze(1)
        if self.opt.scale_size:
            upl = nn.Upsample(tuple(map(int,self.opt.scale_size)), mode='bilinear')
            self.image = upl(self.image)
            self.label = upl(self.label.unsqueeze(1)).squeeze(1)
        if self.opt.add_noise:
            self.image += 0.3*torch.randn(self.image.shape).cuda()
    def set_input(self, data):
        """Unpack input data and perform necessary pre-processing steps.
        self.image: [batch, 1, img_rows, img_cols]
        self.label: [batch, img_rows, img_cols]
        self.filename: str
        """    
        self.image = torch.stack(tuple(data['data'])).cuda()
        self.label = torch.stack(tuple(data['label'])).cuda()
        self.filename = data['filename']    
        self.set_data_pre()
    def opt_TNet(self):      
        """Optimize Task Net seperately
        """  
        self.set_requires_grad(self.TNet,True)
        self.TNet.train()
        pred = self.TNet.forward(self.image,side_out=False)
        self.optimizer_TNet.zero_grad()
        if self.opt.task == 'syn':
            loss = self.TLoss(pred, self.label.unsqueeze(1))
        else:
            loss = self.TLoss(pred, self.label.long())
        loss.backward()
        self.optimizer_TNet.step()
        return loss.data.item()
    def opt_ganAENet(self):
        """Optimize autoencoder with:
            reconstruction loss (MSE)
            latent discriminator loss 
        Code removed, git checkout commits before 172ca44
        """
        pass
    def opt_AEganNet(self):
        """Optimize latent discriminator
        Code removed, git checkout commits before 172ca44
        """        
        pass
    def opt_AENet(self):
        """Optimize Auto-Encoder seperately
        """
        self.set_requires_grad(self.TNet,False)
        self.set_requires_grad(self.AENet,True)
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.train()
        side_out = self.TNet(self.image,side_out=True)
        side_out_cat = []
        ae_out = []
        self.optimizer_AENet.zero_grad()
        loss = 0
        loss_list = []
        weights = self.opt.__dict__.get('weights', [1]*len(self.AENet))                
        for _ in range(len(self.AENet)):
            if len(self.AENetMatch[_]) == 2:
                # concatenate features from the same level
                side_out_cat.append(torch.cat([side_out[self.AENetMatch[_][0]], \
                                               side_out[self.AENetMatch[_][1]]], dim=1))
            else:
                # use seperate features
                side_out_cat.append(side_out[self.AENetMatch[_][0]])         
            ae_out.append(self.AENet[_](side_out_cat[_],side_out=False))
            level_loss = weights[_]*self.AELoss(ae_out[_], side_out_cat[_])
            loss += level_loss
            loss_list.append(level_loss.data.item())
        loss.backward()
        self.optimizer_AENet.step()
        return loss_list
    def opt_TAENet(self):
        """Optimize Auto-Encoder and Task Net jointly
        """
        pass
    def opt_ANet(self,epoch,stable=False):
        """Optimize Adaptor
        """
        self.set_requires_grad([self.TNet] + self.AENet, False) 
        self.set_requires_grad(self.ANet, True)        
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.eval()        
        self.ANet.train() 
        side_out = self.ANet(self.image, self.TNet, side_out=True)     
        side_out_cat = []
        ae_out = []
        self.optimizer_ANet.zero_grad()
        loss = 0
        loss_list = []
        weights = self.opt.__dict__.get('weights', [1]*len(self.AENet))    
        orthw = self.opt.__dict__.get('orthw', 1)  
        if stable:
            loss += self.AELoss(side_out[0],self.image)
            loss_list.append(loss.data.item())      
        else:        
            for _ in range(len(self.AENet)):
                """Keep this part the same with opt_AENet"""
                if len(self.AENetMatch[_]) == 2:
                    # concatenate features from the same level
                    side_out_cat.append(torch.cat([side_out[self.AENetMatch[_][0]], \
                                                side_out[self.AENetMatch[_][1]]], dim=1))
                else:
                    # use seperate features
                    side_out_cat.append(side_out[self.AENetMatch[_][0]])                       
                ae_out.append(self.AENet[_](side_out_cat[_],side_out=False))
                level_loss = weights[_]*self.AELoss(ae_out[_], side_out_cat[_])
                loss += level_loss
                loss_list.append(level_loss.data.item())       
            org_loss = orthw*l2_reg_ortho(self.ANet.conv)   
            loss += org_loss
            loss_list.append(org_loss.data.item())
        loss.backward()
        self.optimizer_ANet.step()
        return loss_list
    def plot(self,image,ids):
        """Plot and save image
        Args:
            image: [channel, img_rows, img_cols] for segmentation masks
                   [img_rows, img_cols] for synthesis
        """
        os.makedirs(os.path.join(self.opt.results_dir,'image'), exist_ok=True)
        ext = ids.split('.')[-1]
        if ext == 'tif':        
            if len(image.shape) > 2:
                image = np.argmax(image,axis=0)
            tiff.imsave(os.path.join(self.opt.results_dir,'image',ids), image)
        else:
            fig = plt.figure(1)
            height = float(image.shape[-2])
            width = float(image.shape[-1])        
            fig.set_size_inches(width/height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if len(image.shape) > 2:
                ax.imshow(np.argmax(image,axis=0))
            else:
                ax.imshow(image,cmap='gray')
            fig.savefig(os.path.join(self.opt.results_dir,'image',ids),dpi=height)
            plt.close(fig)
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
            if self.opt.task == 'syn':
                loss = self.TLoss(pred, self.label.unsqueeze(1))
            else:
                loss = self.TLoss(pred, self.label.long())
        if self.opt.saveimage:
            batch_size = self.image.shape[0]       
            for b_ix in range(batch_size):
                ids = os.path.split(self.filename[b_ix])[-1].split('.')[0]
                self.plot(tonp(pred[b_ix]), ids + '_pred.png')
                self.plot(tonp(self.image[b_ix]), ids + '_image.png')
                self.plot(tonp(self.label[b_ix]), ids + '_label.png')
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
        # save LGan
        if self.opt.lgan and self.opt.trainer != 'tnet':         
            save_path = os.path.join(self.opt.results_dir, 'checkpoints', 
                                     'lcgan' + '_checkpoint_e%d.pth' % (epoch + 1))               
            save_dicts['state_dict'] = self.LGan.cpu().state_dict()
            save_dicts['optimizer'] = self.optimizer_LGan.state_dict()  
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
        elif name is 'lgan':
            self.LGan.load_state_dict(state_dict)  
            self.optimizer_LGan.load_state_dict(optimizer)
            logger.info('LGan weights loaded')            
        else:
            raise TypeError('Unknown model name')

