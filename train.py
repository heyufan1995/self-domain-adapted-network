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
import time
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
parser.add_argument('--sepochs', default=0, type=int, metavar='N',
                    help='number of total epochs to pre-train')                   
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
parser.add_argument('--sub_name', dest='sub_name', default='',type=str,
                    help='path to the txt file name containing subject unique ID')                    
parser.add_argument('--split',dest='split', type=lambda x: list(map(int, x.split(','))),
                    help='the start and end index for validation dataset')
parser.add_argument('--seq',dest='seq', type=lambda x: list(map(int, x.split(','))),
                    help='the 1x1 conv seq to be used in A-Net')
parser.add_argument('--wt',dest='weights', type=lambda x: list(map(float, x.split(','))),
                    help='weights in training ae net') 
parser.add_argument('--wo',dest='orthw', default=1, type=float,
                    help='orthogonal weights in training ANet') 
parser.add_argument('--ps',dest='pad_size', type=lambda x: list(map(float, x.split(','))),
                    help='padding all the input image to this size')     
parser.add_argument('--scs',dest='scale_size', type=lambda x: list(map(float, x.split(','))),
                    help='interpolate all the input image to this size')       
parser.add_argument('--an', dest='add_noise', action='store_true',
                    help = 'add gaussian noise in preprocessing') 
parser.add_argument('--na', dest='no_adpt', action='store_true',
                    help = 'not using pre-adaptation')                                                                                          
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
    # val_loader is a list of dataloader for a list of test subject
    train_loader, val_loader = create_dataset(args)
    logger.info('build dataset done')
    # build model
    model = create_model(args)
    logger.info('build model done')
    # logger.info(model)   
    # evaluate
    if args.evaluate:
        logger.info('begin evaluation')
        loss = validate(model, val_loader[0], args, 0, logger)
        return loss
    if args.test:
        # put test image in vimg_path 
        logger.info('begin testing')
        # train adaptor
        metric_adp, metric_nadp = [], []
        for sub in range(len(val_loader)):
            logger.info('testing subject:{}/{}'.format(sub+1,len(val_loader)))
            # for each test subject, reset optimizer/ANet weights            
            model.ANet.reset()
            model.set_opt()
            prev_loss = np.inf
            sub_metric_adp, sub_metric_nadp = [], []
            start_time = time.time()
            # stablize training by pre-train histogram manipulator (if needed)
            for epoch in range(args.sepochs):  
                m_loss = 0                                       
                for iters, data in enumerate(val_loader[sub]):               
                    model.set_input(data)
                    if iters == 0:
                        logger.info('using stable on subject {}'.format(model.filename))
                    loss = model.opt_ANet(epoch,stable=True)
                    logger.info('[{}/{}][{}/{}] stable Loss: {}'.format(\
                                epoch+1, args.sepochs, iters, len(val_loader[sub]), loss))  
                    m_loss += np.sum(loss)/len(val_loader[sub])
                logger.info('[%d/%d] Mean Loss: %.5f' % (epoch+1, args.tepochs, m_loss)) 
            model.set_opt()
            for epoch in range(args.tepochs):  
                m_loss = 0                                       
                for iters, data in enumerate(val_loader[sub]):               
                    model.set_input(data)
                    if iters == 0:
                        logger.info('subject name {}'.format(model.filename))
                    loss = model.opt_ANet(epoch)
                    logger.info('[{}/{}][{}/{}] Adaptor Loss: {}'.format(\
                                epoch+1, args.tepochs, iters, len(val_loader[sub]), loss))  
                    m_loss += np.sum(loss)/len(val_loader[sub])
                logger.info('[%d/%d] Mean Loss: %.5f' % (epoch+1, args.tepochs, m_loss)) 
                if 0.95*prev_loss < m_loss: 
                    break                    
                else: 
                    prev_loss = m_loss  
            logger.info('training time:{}'.format(time.time()-start_time))    
            # start testing
            logger.info('starting inference')   
            start_time = time.time()         
            for iters, data in enumerate(val_loader[sub]):
                logger.info('[%d/%d]' % (iters, len(val_loader[sub])))
                model.set_input(data)
                _metric = model.test()  
                metric_adp.extend(_metric[0])
                metric_nadp.extend(_metric[1])
                sub_metric_adp.extend(_metric[0])
                sub_metric_nadp.extend(_metric[1])                
                logger.info('metric adp/noadp:\n{}\n{}'\
                      .format(str(_metric[0]).replace('\n',''),\
                              str(_metric[1]).replace('\n','')))
            sub_metric_adp, sub_metric_nadp = np.vstack(sub_metric_adp), np.vstack(sub_metric_nadp)
            logger.info('sub {} mean metric adp/noadp\n{}\n{}'.format(sub+1, \
                         str(np.mean(sub_metric_adp,axis=0)).replace('\n',''),\
                         str(np.mean(sub_metric_nadp,axis=0)).replace('\n','')))  
            logger.info('testing time:{}'.format(time.time()-start_time))     
        metric_adp, metric_nadp = np.vstack(metric_adp), np.vstack(metric_nadp)
        logger.info('Overall mean metric adp/noadp:\n{}[{}]\n{}[{}]'.\
                    format(str(np.mean(metric_adp,axis=0)).replace('\n',''),\
                           np.mean(metric_adp),\
                           str(np.mean(metric_nadp,axis=0)).replace('\n',''),\
                           np.mean(np.vstack(metric_nadp))))
        # there is a "\n" in numpy array which needs to be removed
        with open(os.path.join(args.results_dir, args.task+'_adp.txt'),'w') as f:
            f.writelines(["%s\n" % str(item).replace('\n','') for item in metric_adp])
        with open(os.path.join(args.results_dir, args.task+'_noadp.txt'),'w') as f:
            f.writelines(["%s\n" % str(item).replace('\n','') for item in metric_nadp])  
        with open(os.path.join(args.results_dir, args.task+'_args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)              
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
                            epoch+1, args.epochs, iters, len(train_loader),
                            args.trainer, loss))                                     
                m_loss += np.sum(loss)/len(train_loader)
            logger.info('[%d/%d] %s Mean Loss: %.5f' % (epoch+1, args.epochs, 
                                                        args.trainer, m_loss))      
            save_path = os.path.join(args.results_dir, args.trainer+'_train_history.csv')
            with open(save_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1,m_loss])   
            if (epoch+1) % args.save_step == 0 or epoch+1 == args.epochs:
                if args.trainer == 'tnet':
                    loss = validate(model, val_loader[0], args, epoch, logger)
                model.save_nets(epoch)

def validate(model, val_loader, args, epoch, logger):
    m_loss = 0
    for iters, data in enumerate(val_loader):
        model.set_input(data)
        loss = model.eval()  
        logger.info('[%d/%d][%d/%d] Loss: %.5f' % \
                (epoch+1, args.epochs, iters, len(val_loader), loss))            
        m_loss += loss/len(val_loader)
    logger.info('[%d/%d] Mean Loss: %.5f' % (epoch+1, args.epochs, m_loss))                      
    return m_loss

if __name__ == '__main__':
    main()

