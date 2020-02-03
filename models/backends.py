import functools
import torch
import torch.nn.functional as F
import functools
import torch.nn as nn
import math
import pdb
import logging
import numpy as np
class _ConvBlock(nn.Module):
    def __init__(self, inplane, outplane, kernel=(3,3), **kwargs):
        super(_ConvBlock,self).__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        pad = kwargs.get('pad', None)    
        if pad is None:
            pad = (kernel[0]//2, kernel[1]//2)            
        self.conv = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel, padding=pad),
            nn.BatchNorm2d(outplane),
            nn.ReLU(inplace=True),            
            #nn.Dropout2d(p=0.2),
            nn.Conv2d(outplane, outplane, kernel, padding=pad),
            nn.BatchNorm2d(outplane),
            nn.ReLU(inplace=True)
            #nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        x = self.conv(x) 
        return x

class _ResConvBlock(nn.Module):
    def __init__(self, inplane, outplane, kernel=(3,3), **kwargs):
        super(_ResConvBlock,self).__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        pad = kwargs.get('pad', None)
        gpn = kwargs.get('gpn', False)
        isn = kwargs.get('isn', False)
        reflect = kwargs.get('reflect', False)
        stride = kwargs.get('stride',1)
        if pad is None:
            pad = (kernel[0]//2, kernel[1]//2)
        # use reflect padding to solve the boundary problem
        if reflect:
            self.reflectpad1 = nn.ReflectionPad2d((pad[0], pad[0], pad[1], pad[1]))
            self.reflectpad2 = nn.ReflectionPad2d((2*pad[0], 2*pad[0], 2*pad[1], 2*pad[1]))
            pad = (0, 0)
        self.conv = nn.Conv2d(inplane, outplane, kernel, stride=stride, padding=pad)
        # use group norm instead of batch norm 
        if gpn:
            self.norm = functools.partial(nn.GroupNorm,4) 
        elif isn:
            self.norm = nn.InstanceNorm2d
        else:
            self.norm = nn.BatchNorm2d
        self.resconv = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel, padding=pad),
            self.norm(outplane),
            nn.PReLU(),            
            #nn.Dropout2d(p=0.2),
            nn.Conv2d(outplane, outplane, kernel, padding=pad),
            self.norm(outplane),
            #nn.Dropout2d(p=0.2)
        )
        self.relu = nn.PReLU()
        
    def forward(self, x):       
        if hasattr(self,'reflectpad1'):
            x = self.reflectpad1(x)
        residual = self.conv(x)        
        if hasattr(self,'reflectpad2'):
            temp = self.reflectpad2(residual)
        else:
            temp = residual.clone()
        x = self.resconv(temp) + residual
        x = self.relu(x)
        return x  

ConvBlock = _ResConvBlock


class UpBlock(nn.Module):
    def __init__(self, inplane, outplane, kernel=(3,3), **kwargs):
        super(UpBlock,self).__init__()       
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)     
        self.skip=kwargs.get('skip', True)             
        self.uconv = ConvBlock(inplane, outplane, kernel, **kwargs)

    def forward(self, xl, xh=None):  
        if xh is not None: 
            x = F.upsample(xl,size=xh.shape[-2:],mode='bilinear',align_corners=True)         
            if self.skip:            
                x = torch.cat([x, xh], dim=1)
        else:
            x = self.up(xl)
        x = self.uconv(x)
        return x 

class DownBlock(nn.Module):
    def __init__(self, inplane, outplane, kernel=(3,3), **kwargs):
        super(DownBlock,self).__init__()
        stride = kwargs.get('down_stride',None)
        if stride:
            kwargs['stride'] = stride
            self.dconv = ConvBlock(inplane, outplane, kernel, **kwargs)
        else:
            self.dconv = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(inplane, outplane, kernel, **kwargs)
            )
    def forward(self, x):
        x = self.dconv(x)
        return x         

class UNet(nn.Module):
    def __init__(self, inplane, midplane, outplane, 
                 kernel=(3,3), **kwargs):
        super(UNet,self).__init__()
        self.skip=kwargs.get('skip', True)
        self.inplane = inplane
        self.midplane = midplane
        self.outplane = outplane
        self.depth = len(midplane) -1
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.inblocks = ConvBlock(self.inplane, self.midplane[0], kernel, **kwargs)
        # allow user defined bottleneck
        self.bottleneck = kwargs.get('bottleneck', \
                          ConvBlock(self.midplane[-1], self.midplane[-1], 1, **kwargs))
        self.outblock = nn.Conv2d(self.midplane[0], self.outplane, 1)
        for i in range(self.depth):
            self.downblocks.append(DownBlock(self.midplane[i], 
                                             self.midplane[i+1],
                                             kernel, **kwargs))
            self.upblocks.append(UpBlock(self.skip*self.midplane[i] + \
                                         self.midplane[i+1],\
                                         self.midplane[i], 
                                         kernel, **kwargs))
        
    def forward(self,x,side_out=False,bot_out=False):
        xh = [x]  
        x = self.inblocks(x)
        xh.append(x)
        for i in range(self.depth):
            x = self.downblocks[i](x)
            xh.append(x)
        x = self.bottleneck(x)
        xh.append(x)
        for i in range(self.depth):
            x = self.upblocks[self.depth-i-1](x,xh[self.depth-i])
            xh.append(x)
        x = self.outblock(x)
        xh.append(x)
        if side_out:
            return xh
        else:
            return x

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, dilations=1):
        super().__init__()
        self.add_module('norm', nn.InstanceNorm2d(in_channels))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=dilations,
                                          dilation=dilations, bias=True))        
        self.add_module('relu', nn.PReLU())
    def forward(self, x):
        return super().forward(x)
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, dilations=None, upsample=False):
        super().__init__()
        if dilations == None:
            dilations = [1]*n_layers
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate, dilations=dilations[i])
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x