from __future__ import division
import matplotlib
from matplotlib import pyplot as plt
from scipy.io import savemat
import pdb
import argparse
import functools
import warnings
import logging
logging.basicConfig(level=logging.INFO)
import os
import shutil
import time
from scipy import misc
import numpy as np
import json,csv
import time
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim

from datasets import create_dataset
from models import create_model 
from utils.util import setlogger, deterministic
from config import *

def main():
    warnings.filterwarnings('ignore')
    # set random seed
    deterministic()
    args = parser.parse_args()
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
        val_loader[0].dataset.augment = False
        loss = validate(model, val_loader[0], args, 0, logger)
        return loss
    if args.test:
        # put test image in vimg_path 
        logger.info('begin testing')
        # train adaptor
        metric_adp, metric_nadp = [], []
        metric_adp3d, metric_nadp3d = [], []
        for sub in range(len(val_loader)):
            logger.info('testing subject:{}/{}'.format(sub+1,len(val_loader)))
            model.ANet.reset()
            val_loader[sub].dataset.augment = args.val_augment
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
                if prev_loss < m_loss: 
                    break                    
                else: 
                    prev_loss = m_loss 
            logger.info('training time:{}'.format(time.time()-start_time))    
            # start testing
            logger.info('starting inference')   
            start_time = time.time()   
            # turn off augmentation on test inference
            val_loader[sub].dataset.augment = False 
            # allow 3D metric calculation
            label3d, pred3d, pred_na3d = [], [], []                  
            for iters, data in enumerate(val_loader[sub]):
                logger.info('[%d/%d]' % (iters, len(val_loader[sub])))
                model.set_input(data)
                _metric, pred, pred_na = model.test(return_pred=True)  
                metric_adp.extend(_metric[0])
                metric_nadp.extend(_metric[1])
                sub_metric_adp.extend(_metric[0])
                sub_metric_nadp.extend(_metric[1])                
                logger.info('metric adp/noadp:\n{}\n{}'\
                      .format(str(_metric[0]).replace('\n',''),\
                              str(_metric[1]).replace('\n','')))
                label3d.append(model.label)
                pred3d.append(pred)
                pred_na3d.append(pred_na)
            label3d = torch.stack(label3d)
            pred3d = torch.stack(pred3d)
            pred_na3d = torch.stack(pred_na3d)
            _metric3d_adp = model.cal_metric3d(pred3d.view(-1,pred3d.shape[-3],pred3d.shape[-2], pred3d.shape[-1]),\
                                               label3d.view(-1,label3d.shape[-2], label3d.shape[-1]))
            _metric3d_nadp = model.cal_metric3d(pred_na3d.view(-1,pred_na3d.shape[-3],pred_na3d.shape[-2], pred_na3d.shape[-1]), \
                                                label3d.view(-1,label3d.shape[-2], label3d.shape[-1]))
            metric_adp3d.append(_metric3d_adp)
            metric_nadp3d.append(_metric3d_nadp)
            sub_metric_adp, sub_metric_nadp = np.vstack(sub_metric_adp), np.vstack(sub_metric_nadp)
            logger.info('sub {} mean metric adp/noadp\n{}\n{}'.format(sub+1, \
                         str(np.nanmean(sub_metric_adp,axis=0)).replace('\n',''),\
                         str(np.nanmean(sub_metric_nadp,axis=0)).replace('\n','')))  
            logger.info('sub {} mean 3D metric adp/noadp\n{}\n{}'.format(sub+1, \
                         str(_metric3d_adp).replace('\n',''),\
                         str(_metric3d_nadp).replace('\n','')))  
            logger.info('testing time:{}'.format(time.time()-start_time))     
        metric_adp, metric_nadp = np.vstack(metric_adp), np.vstack(metric_nadp)
        metric_adp3d, metric_nadp3d = np.vstack(metric_adp3d), np.vstack(metric_nadp3d)
        logger.info('Overall mean metric adp/noadp:\n{}[{}]\n{}[{}]'.\
                    format(str(np.nanmean(metric_adp,axis=0)).replace('\n',''),\
                           np.nanmean(metric_adp),\
                           str(np.nanmean(metric_nadp,axis=0)).replace('\n',''),\
                           np.nanmean(np.vstack(metric_nadp))))
        logger.info('Overall 3D mean metric adp/noadp:\n{}[{}]\n{}[{}]'.\
                    format(str(np.nanmean(metric_adp3d,axis=0)).replace('\n',''),\
                           np.nanmean(metric_adp3d),\
                           str(np.nanmean(metric_nadp3d,axis=0)).replace('\n',''),\
                           np.nanmean(np.vstack(metric_nadp3d))))
        # there is a "\n" in numpy array which needs to be removed
        with open(os.path.join(args.results_dir, args.task+'_adp.txt'),'w') as f:
            f.writelines(["%s\n" % str(item).replace('\n','') for item in metric_adp])
        with open(os.path.join(args.results_dir, args.task+'_noadp.txt'),'w') as f:
            f.writelines(["%s\n" % str(item).replace('\n','') for item in metric_nadp])  
        with open(os.path.join(args.results_dir, args.task+'_adp3d.txt'),'w') as f:
            f.writelines(["%s\n" % str(item).replace('\n','') for item in metric_adp3d])
        with open(os.path.join(args.results_dir, args.task+'_noadp3d.txt'),'w') as f:
            f.writelines(["%s\n" % str(item).replace('\n','') for item in metric_nadp3d])  
        with open(os.path.join(args.results_dir, args.task+'_args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)              
        return   
    # train tnet or aenet
    best_loss, best_epoch = np.inf, 0
    with open(os.path.join(args.results_dir, args.trainer+'_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)           
    for epoch in range(args.epochs):
        m_loss = 0       
        for iters, data in enumerate(train_loader):
            model.set_input(data)
            # plt.imshow(data['data'][0,0],cmap='gray');plt.show();plt.savefig('{}_i'.format(iters))
            # plt.imshow(data['label'][0]);plt.show();plt.savefig('{}_l'.format(iters))
            if args.trainer == 'tnet':
                loss = model.opt_TNet()
            else:
                if 'seg' in args.task and ((data['label'] > 0).sum(1).sum(1) == 0 ).any():
                    logger.info('skip background')
                    loss = 0
                    continue
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
                loss = validate(model, val_loader[0], args, epoch+1, logger)
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch + 1
            model.save_nets(epoch+1)
            logger.info('best loss {} at epoch {}'.format(best_loss, best_epoch))

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

