from datasets.dataset import PairDataset,SegDataset
from datasets.dataset import split_data
from torch.utils.data import Dataset, DataLoader
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
    if args.vimg_path:          
        val_dataset = dataset(filepath=args.vimg_path, 
                                 labelpath=args.vlabel_path,
                                 file_ext=args.img_ext,
                                 lab_ext=args.label_ext)
    elif args.split:
        train_dataset, val_dataset = split_data(dataset=train_dataset,
                                                split=args.split)
        train_dataset = train_dataset
        val_dataset = val_dataset
    else:
        # no validation used
        val_dataset = dataset()
    # transform
    # pass
    # normalize
    # pass
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers, 
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True)
    return train_loader, val_loader