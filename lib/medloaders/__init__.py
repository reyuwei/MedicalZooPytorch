from torch.utils.data import DataLoader
from .mrihand import MRIHandDataset
import os
import numpy as np


def generate_datasets(args, path='.././datasets', params=None):
    if params is None:
        train_params = {'batch_size': args.batchSz,
                'shuffle': True,
                'num_workers': args.worker}
        val_prams = {'batch_size': args.batchSz,
                'shuffle': False,
                'num_workers': args.worker}

    if args.dataset_name == "mrihand":
        split_pkl = os.path.join(path, "splits_final.pkl")
        split = np.load(split_pkl, allow_pickle=True)[0]
        train_lst = split['train']
        val_lst = split['val']
        if args.segonly:
            crop_input=True
        if args.jointonly:
            crop_input=False
        train_loader = MRIHandDataset(args, 'train', dataset_path=path, crop_dim=args.dim,
                                          lst=train_lst, load=args.loadData, crop_input=crop_input)
        val_loader = MRIHandDataset(args, 'val', dataset_path=path, crop_dim=args.dim, 
                                           lst=val_lst, load=args.loadData, crop_input=crop_input)

    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **val_prams)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine