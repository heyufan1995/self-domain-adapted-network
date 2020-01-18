from __future__ import division
import matplotlib
from matplotlib import pyplot as plt
from scipy.io import savemat
import pdb
import argparse
import functools
import logging
logging.basicConfig(level=logging.INFO)
import os
import shutil
import time
from scipy import misc
import numpy as np
np.random.seed(0)
import json,csv

import torch
torch.manual_seed(0)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim

from datasets import create_dataset
from models import create_model 
from utils.util import setlogger
parser = argparse.ArgumentParser(description='PyTorch parser')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--tepochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')                    
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--tlr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for TNet')
parser.add_argument('--aelr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for AENet')
parser.add_argument('--alr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for ANet')  
parser.add_argument('--llr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for LGan')                                                          
parser.add_argument('--resume_T', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_AE', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)') 
parser.add_argument('--resume_LG', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')                                       
parser.add_argument('-p','--pretrain', dest='pretrain', action='store_true',
                    help='use the pretrain model and reset optimizer')
parser.add_argument('--trainer',dest='trainer', default='tnet', 
                    type=str, help='select which net to train')  
parser.add_argument('--task',dest='task', default='syn', 
                    type=str, help='select task (syn:synthesis or seg:segmentation)')                   
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='test model on validation set')                    
parser.add_argument('--label_path', dest='label_path', default='../data/label/',type=str,
                    help='path to the label')
parser.add_argument('--label_ext', dest='label_ext', default='json',type=str,
                    help='label extension')
parser.add_argument('--img_path', dest='img_path', default='../data/image/',type=str,
                    help='path to the image')
parser.add_argument('--img_ext', dest='img_ext', default='png',type=str,
                    help='image extension')
parser.add_argument('--vlabel_path', dest='vlabel_path', default='',type=str,
                    help='path to the validation label')
parser.add_argument('--vimg_path', dest='vimg_path', default='',type=str,
                    help='path to the validation image')
parser.add_argument('--split',dest='split', type=lambda x: list(map(int, x.split(','))),
                    help='the start and end index for validation dataset')
parser.add_argument('--wt',dest='weights', type=lambda x: list(map(float, x.split(','))),
                    help='weights in training ae net') 
parser.add_argument('--ps',dest='pad_size', type=lambda x: list(map(float, x.split(','))),
                    help='padding all the input image to this size')     
parser.add_argument('--scs',dest='scale_size', type=lambda x: list(map(float, x.split(','))),
                    help='interpolate all the input image to this size')                                                        
parser.add_argument('--results_dir', dest='results_dir', default='results_dir',
                    help='results dir of output')
parser.add_argument('--config', dest='config', default='config.json',
                    help='hyperparameter in json format')
parser.add_argument('--ss', dest='save_step', default=5, type=int,
                    help = 'The step of epochs to save checkpoints and validate')
parser.add_argument('--saveimage','--si', dest='saveimage', action='store_true',
                    help='save image with surfaces and layers')
parser.add_argument('--dpi', dest='dpi', type=int, default=100, help='dpi of saved image')
parser.add_argument('--lgan', dest='lgan', action='store_true', help='use gan loss')
def main():
    args = parser.parse_args()
    args.isTrain = False
    os.makedirs(args.results_dir, exist_ok=True)
    logger = logging.getLogger('global')
    logger = setlogger(logger,args)
    # build dataset
    train_loader, val_loader = create_dataset(args)
    logger.info('build dataset done')
    # build model
    model = create_model(args)
    logger.info('build model done')
    # logger.info(model)   
    # evaluate
    if args.evaluate:
        logger.info('begin evaluation')
        loss = validate(model, val_loader, args, 0, logger)
        return loss
    # test
    if args.test:
        # put test image in vimg_path 
        logger.info('begin testing')
        # train adaptor
        for epoch in range(args.tepochs):
            m_loss = 0
            for iters, data in enumerate(val_loader):                
                model.set_input(data)
                loss = model.opt_ANet(epoch)
                logger.info('[{}/{}][{}/{}] Adaptor Loss: {}'.format(\
                            epoch, args.tepochs, iters, len(val_loader), loss))  
                m_loss += np.sum(loss)/len(val_loader)
            logger.info('[%d/%d] Mean Loss: %.5f' % (epoch, args.tepochs, m_loss))   
        # start testing
        logger.info('starting inference')
        metric_adp, metric_nadp = [], []
        for iters, data in enumerate(val_loader):
            logger.info('[%d/%d]' % (iters, len(val_loader)))
            model.set_input(data)
            _metric = model.test()  
            metric_adp.extend(_metric[0])
            metric_nadp.extend(_metric[1])
            logger.info('metric adp/noadp:{}/{}'.format(_metric[0],_metric[1]))
        logger.info('mean metric adp/noadp{}/{}'.format(np.mean(metric_adp),\
                                                        np.mean(metric_nadp)))
        return   
    # train
    if args.lgan and args.trainer != 'tnet':
        '''Train autoencoder with latent gan
           OCGAN: One-class Novelty Detection
        Code removed, git checkout commits before 172ca44
        '''
        pass
    else:
        for epoch in range(args.start_epoch, args.epochs):
            m_loss = 0
            for iters, data in enumerate(train_loader):
                model.set_input(data)
                if args.trainer == 'tnet':
                    loss = model.opt_TNet()
                else:
                    loss = model.opt_AENet()
                logger.info('[{}/{}][{}/{}] {} Loss: {}'.format(\
                            epoch, args.epochs, iters, len(train_loader),
                            args.trainer, loss))                                     
                m_loss += np.sum(loss)/len(train_loader)
            logger.info('[%d/%d] %s Mean Loss: %.5f' % (epoch, args.epochs, 
                                                        args.trainer, m_loss))      
            save_path = os.path.join(args.results_dir, args.trainer+'_train_history.csv')
            with open(save_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1,m_loss])   
            if (epoch+1) % args.save_step == 0 or epoch+1 == args.epochs:
                # loss = validate(model, val_loader, args, epoch, logger)
                model.save_nets(epoch)

def validate(model, val_loader, args, epoch, logger):
    m_loss = 0
    for iters, data in enumerate(val_loader):
        model.set_input(data)
        loss = model.eval()  
        logger.info('[%d/%d][%d/%d] Loss: %.5f' % \
                (epoch, args.epochs, iters, len(val_loader), loss))            
        m_loss += loss/len(val_loader)
    logger.info('[%d/%d] Mean Loss: %.5f' % (epoch, args.epochs, m_loss))                      
    return m_loss

if __name__ == '__main__':
    main()

