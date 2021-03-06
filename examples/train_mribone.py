# Python libraries
import argparse
import os

import torch
import numpy as np
import sys
sys.path.append("/p300/liyuwei/MRI_Bonenet/MedicalZooPytorch")
sys.path.append("F:\\OneDrive\\Projects_ongoing\\10_HANDMRI\\mri_bone_net\\MedicalZooPytorch\\")

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D.JoinLoss import JoinLoss
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 1777777
torch.manual_seed(seed)


# with open('commandline_args.txt', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)

# parser = ArgumentParser()
# args = parser.parse_args()
# with open('commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)

# print(args)


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    print(args)
    with open(args.save + 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(args.save + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    # training_generator, val_generator, full_volume, affine = \
    #                 medical_loaders.generate_datasets(args, path='/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final')
    # training_generator, val_generator, full_volume, affine = \
    #             medical_loaders.generate_datasets(args, path='/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final_t1')
    training_generator, val_generator, full_volume, affine = \
                medical_loaders.generate_datasets(args, path=args.dataset)
    model, optimizer = medzoo.create_model(args)
    criterion = JoinLoss(classes=args.classes, skip_index_after=args.classes)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final')
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="mrihand")
    # parser.add_argument('--dim', nargs="+", type=int, default=(128, 128, 128))
    parser.add_argument('--dim', nargs="+", type=int, default=(64,64,64))
    parser.add_argument('--nEpochs', type=int, default=9999)
    parser.add_argument('--classes', type=int, default=21)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--threshold', default=0.0001, type=float)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    # parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-2, type=float,  help='learning rate (default: 1e-2)')
    parser.add_argument('--split', default=0.7, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--worker', default=0, type=int, help="workers")
    
    parser.add_argument('--segonly', action='store_true', default=False)
    parser.add_argument('--segnet', type=str, default="unet3d")
    parser.add_argument('--joint_center_idx', type=int, default=0)
    parser.add_argument('--use_lbs', action='store_true', default=False)
    parser.add_argument('--encoderonly', action='store_true', default=False)

    parser.add_argument('--loadData', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='MRIBONENET',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')
    parser.add_argument("--gpu", type=str, default="0", help="select gpuid")

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
