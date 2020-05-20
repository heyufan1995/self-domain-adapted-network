from datasets.dataset import PairDataset,SegDataset
from datasets.dataset import split_data
from torch.utils.data import Dataset, DataLoader
import os
import pdb
from copy import deepcopy
import logging
logger = logging.getLogger('global')
def create_dataset(args):
    ''' Create train and val dataset
    '''
    # Data loading code
    if args.task == 'syn':
        dataset = PairDataset
    elif args.task == 'seg':
        dataset = SegDataset
    else:
        raise NotImplementedError
    train_dataset = dataset(filepath=args.img_path, 
                                labelpath=args.label_path,
                                file_ext=args.img_ext, 
                                lab_ext=args.label_ext)
    val_dataset_list = []
    if args.vimg_path:           
        val_dataset = dataset(filepath=args.vimg_path, 
                                 labelpath=args.vlabel_path,
                                 file_ext=args.img_ext,
                                 lab_ext=args.label_ext)
        # return a list of val_dataset, each contains one subject             
        if args.__dict__.get('sub_name',False):
            if os.path.isfile(args.sub_name):
                with open(args.sub_name) as f:
                    dataname = f.read().splitlines() 
            datalist = val_dataset.datalist
            labellist = val_dataset.labellist
            for name in dataname:
                val_dataset.datalist = sorted([_ for _ in datalist if name in str(_)])
                val_dataset.labellist = sorted([_ for _ in labellist if name in str(_)])
                val_dataset_list.append(deepcopy(val_dataset))
        else:
            val_dataset_list = [val_dataset]


    elif args.split and not args.__dict__.get('sub_name',False):
        train_dataset, val_dataset = split_data(dataset=train_dataset,
                                                split=args.split,
                                                switch=args.test or args.evaluate)
        train_dataset = train_dataset
        val_dataset = val_dataset
        val_dataset_list = [val_dataset]
    else:
        # no validation used
        val_dataset = dataset()
        val_dataset_list = [val_dataset]
    # transform
    # pass
    # normalize
    # pass
    # check lablelist is non-empty
    if sum([len(_.labellist) == 0 for _ in val_dataset_list]):
        logger.warning('None-exist label: use data as label')
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers, 
                              pin_memory=True)
    val_loader = [DataLoader(dataset=_,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.workers,
                            pin_memory=True) for _ in val_dataset_list]
    return train_loader, val_loader