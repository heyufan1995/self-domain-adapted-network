import os
import json
import logging
import numpy as np
import torch
def grams(x):
    """Return gram matrix for input feature
    Args:
        x: feature maps, [batch, channel, rows, cols]
    Return:
        grams [batch, channel, channel]
    """
    a, b, c, d = x.size() 
    features = x.view(a * b, c * d) 
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)

def load_config(config_path,args):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    logger = logging.getLogger('global')
    logger.info(json.dumps(cfg, indent=2))
    save_path = os.path.join(args.results_dir,
                             os.path.basename(args.config))
    if not args.evaluate:
        with open(save_path, 'w') as fp:
            fp.write(json.dumps(cfg, indent=2))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg

def setlogger(logger,args):
    if args.test or args.evaluate:
        hdlr = logging.FileHandler(os.path.join(args.results_dir,'eval.log'),'w+')
    else:
        if args.resume_T or args.resume_AE:
            hdlr = logging.FileHandler(os.path.join(args.results_dir,args.trainer+'_train.log'))
        else:
            hdlr = logging.FileHandler(os.path.join(args.results_dir,args.trainer+'_train.log'),'w+')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info(json.dumps(vars(args), indent=2))
    return logger
    
def ncc(img1, img2):
    """Normalized cross correlation loss
    """
    return 1 - 1.*torch.sum((img1 - torch.mean(img1,dim=[-1,-2]))*(img2 - torch.mean(img2,dim=[-1,-2])))\
            /torch.sqrt(torch.sum((img1 - torch.mean(img1,dim=[-1,-2]))**2) \
            *torch.sum((img2 - torch.mean(img2,dim=[-1,-2]))**2) + 1e-10)

def generate_mask(bd_pts=None, img_rows=None, lesion=None):
    """ Generate masks from boundary surfaces and lesion masks

        Args:
            bd_pts: boundary surfaces, shape = [bds, img_cols]
            lesion: lesion masks, shape =  [img_rows, img_cols] 

        Returns:
            label: training labels, shape = (img_rows, img_cols)

    """
    bds,img_cols = bd_pts.shape
    if img_rows is None:
        assert lesion is not None
        img_rows = lesion.shape[0]
    label = np.zeros((img_rows,img_cols))*np.nan
    for j in range(img_cols):
        
        cols = np.arange(img_rows)
        index = cols - bd_pts[0,j] < 0
        label[index,j] = 0
        index = cols - bd_pts[bds-1,j] >= 0
        label[index,j] = bds
        for k in range(bds-1):
            index_up = cols - bd_pts[k,j] >= 0
            index_down = cols - bd_pts[k+1,j] < 0
            label[index_up&index_down,j] = k+1
    if lesion is not None:
        label[lesion>0] = bds+1

    return label