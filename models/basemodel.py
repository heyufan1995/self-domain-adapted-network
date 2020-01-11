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

class ANet(nn.Module):
    def __init__(self):
        super(ANet,self).__init__()
        self.conv = nn.ModuleList()
        eye = nn.init.eye_(torch.empty(64, 64)).unsqueeze(-1).unsqueeze(-1)
        for _ in range(6):
            convs = nn.Conv2d(64,64,1)
            convs.weight.data = eye
            self.conv.append(convs)
    def forward(self, x, TNet, side_out=False):
        ct = 0
        xh = [x]  
        x = TNet.inblocks(x)
        x = self.conv[ct](x)
        ct += 1
        xh.append(x)
        for i in range(TNet.depth):
            x = TNet.downblocks[i](x)            
            if i != TNet.depth - 1:
                x = self.conv[ct](x)
                ct += 1            
            xh.append(x)
        x = TNet.bottleneck(x)
        xh.append(x)
        for i in range(TNet.depth):
            x = TNet.upblocks[TNet.depth-i-1](x,xh[TNet.depth-i])
            x = self.conv[ct](x)
            ct += 1            
            xh.append(x)
        x = TNet.outblock(x)
        xh.append(x)
        if side_out:
            return xh
        else:
            return x 

class AdaptorNet(nn.Module):
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
        """
        self.AENet = [UNet(128,[64,32,16],128,isn=True,skip=False),
                      UNet(128,[64,32,16],128,isn=True,skip=False),
                      UNet(128,[64,32,16],128,isn=True,skip=False)]
        self.AENetMatch = [[1,-2],[2,-3],[3,-4]] # the matching index of TNet features
    def def_ANet(self):
        """Define Adaptor Net for domain adapt
        """
        self.ANet = ANet()
    def def_LGan(self):
        """Define latent space discriminator
        """
        inplane = 32
        self.LGan = nn.Sequential(
            nn.Linear(inplane,inplane//2),
            nn.LeakyReLU(),
            nn.Linear(inplane//2,inplane//4),
            nn.LeakyReLU(),
            nn.Linear(inplane//4,inplane//8)       
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
    def opt_ganAENet(self):
        """Optimize autoencoder with:
            reconstruction loss (MSE)
            latent discriminator loss 
            visualization discriminator loss
        """
        self.set_requires_grad([self.TNet,self.LGan],False)  
        self.set_requires_grad(self.AENet,True)
        self.TNet.eval()
        self.LGan.eval()
        for subnets in self.AENet:
            subnets.train()
        side_out = list(map(self.TNet(self.image,side_out=True).__getitem__, self.AENetMatch))     
        ae_out = [self.AENet[_](side_out[_],side_out=True) for _ in range(len(self.AENet))]
        self.optimizer_AENet.zero_grad()      
        # reconstruction loss
        rec_loss = 0
        weights = self.opt.__dict__.get('weights', [1]*len(ae_out))
        for _ in range(len(ae_out)):
            rec_loss += weights[_]*self.AELoss(ae_out[_][-1], side_out[_]) 
        # lgan loss: make the latent space indistinguishable from U[-1,1]
        # concatenate bottleneck output from AE
        cat_feat = []
        for _ in range(len(ae_out)):
            cat_feat.append(ae_out[_][self.botindex])
        cat_feat = torch.cat(cat_feat, dim=1)
        fake_l = F.tanh(F.adaptive_avg_pool2d(cat_feat,1)).squeeze(-1).squeeze(-1)
        fake_c = self.LGan(fake_l)
        gan_loss = self.LGanLoss(fake_c,\
                    torch.FloatTensor(fake_c.data.size()).fill_(1).cuda())
        loss = gan_loss + rec_loss
        loss.backward()
        self.optimizer_AENet.step()
        return [rec_loss.data.item(), gan_loss.data.item()]
    def opt_AEganNet(self):
        """Optimize latent discriminator
        """        
        self.set_requires_grad([self.TNet] + self.AENet,False)  
        self.set_requires_grad(self.LGan,True)
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.eval()
        self.LGan.train()
        side_out = list(map(self.TNet(self.image,side_out=True).__getitem__, self.AENetMatch))     
        ae_out = [self.AENet[_](side_out[_],side_out=True) for _ in range(len(self.AENet))]
        self.optimizer_LGan.zero_grad()
        cat_feat = []
        for _ in range(len(ae_out)):
            cat_feat.append(ae_out[_][self.botindex])
        cat_feat = torch.cat(cat_feat, dim=1)
        fake_l = F.tanh(F.adaptive_avg_pool2d(cat_feat,1)).squeeze(-1).squeeze(-1)
        fake_c = self.LGan(fake_l)
        real_l = torch.tensor(np.random.uniform(-1,1,fake_l.data.size()), \
                              dtype=torch.float32, device=fake_l.device)
        real_c = self.LGan(real_l)
        loss = self.LGanLoss(fake_c,\
                    torch.FloatTensor(fake_c.data.size()).fill_(0).cuda()) \
             + self.LGanLoss(real_c,\
                    torch.FloatTensor(real_c.data.size()).fill_(1).cuda())
        loss.backward()
        self.optimizer_LGan.step()
        return loss.data.item()       
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
        weights = self.opt.__dict__.get('weights', [1]*len(ae_out))        
        for _ in range(len(self.AENet)):
            side_out_cat.append(torch.cat([side_out[self.AENetMatch[_][0]], \
                                           side_out[self.AENetMatch[_][1]]], dim=1)) 
            ae_out.append(self.AENet[_](side_out_cat[_]))
            loss += weights[_]*self.AELoss(ae_out[_], side_out_cat[_])             
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
        weights = self.opt.__dict__.get('weights', [1]*len(ae_out))      
          
        # for _ in range(len(self.AENet)):
        for _ in range(1):
            side_out_cat.append(torch.cat([side_out[self.AENetMatch[_][0]], \
                                           side_out[self.AENetMatch[_][1]]], dim=1)) 
            ae_out.append(self.AENet[_](side_out_cat[_]))
            loss += weights[_]*self.ALoss(ae_out[_], side_out_cat[_])     
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

