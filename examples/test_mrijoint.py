# Python libraries
import argparse
import os
import SimpleITK as sitk
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
from lib.train.trainer import prepare_input
from lib.losses3D.JoinLoss import JoinLoss
from lib.losses3D.PayerLoss import PayerLoss, project_3d_joint
import json
seed = 1777777
torch.manual_seed(seed)

def main():
    
    args = argparse.ArgumentParser()
    args = args.parse_args()
    parserfile = "../saved_models/MRIJOINTNET_checkpoints/MRIJOINTNET_07_02___06_53_mrihand_/args.txt" #unet

    with open(parserfile, 'r') as f:
        args.__dict__ = json.load(f)
    print(args)
    args.resume = args.save
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    args.worker =0
    args.batchSz = 1
    args.encoderonly = False
    args.shuffle=False
    params = {'batch_size': 1,
        'shuffle': False,
        'num_workers': 1}
    training_generator, val_generator, full_volume, affine = \
                medical_loaders.generate_datasets(args, path=args.dataset)
    model, optimizer = medzoo.create_model(args)
    criterion = PayerLoss(joints=args.joints, heatmap_size=args.dim, sigma_scale=args.sigma_scale, weight_heatmap=1, weight_sigma=1)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    os.makedirs(args.save + "/eval", exist_ok=True)
    meta = {}

    trainer.model.eval()
    meta['joint_error'] = []
    for batch_idx, input_tuple in enumerate(trainer.valid_data_loader):
        with torch.no_grad():
            print(batch_idx, len(trainer.valid_data_loader))
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=trainer.args)
            pred = trainer.model(input_tensor)
            
            target_landmark, affine = target
            pred_heatmap, sigmas = pred
            sigmas = torch.ones_like(sigmas) * 2.5
            target_landmark_idx = project_3d_joint(target_landmark, affine)
            target_heatmap = criterion.generate_heatmap(target_landmark_idx, sigmas)
            joint_error = criterion.joint_error(pred_heatmap, target_landmark_idx)

            meta['joint_error'].append(float(joint_error.detach().cpu().numpy()))

            sitk.WriteImage(sitk.GetImageFromArray(pred_heatmap.squeeze().detach().cpu().numpy().transpose(1,2,3,0)), args.save + "/eval" + "/{:02d}_hmo.nii".format(batch_idx))
            sitk.WriteImage(sitk.GetImageFromArray(target_heatmap.squeeze().detach().cpu().numpy().transpose(1,2,3,0)), args.save + "/eval" + "/{:02d}_hmgt.nii".format(batch_idx))
            sitk.WriteImage(sitk.GetImageFromArray(input_tensor.squeeze().detach().cpu().numpy()), args.save + "/eval" + "/{:02d}_input.nii".format(batch_idx))

    meta['mean_joint_error'] = np.mean(np.stack(meta['joint_error']))

    print(meta)
    with open(args.save + "/eval" + "/metric.json", 'w') as f:
        json.dump(meta, f, indent=2)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final')
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="mrihand")
    parser.add_argument('--dim', nargs="+", type=int, default=(128, 128, 128))
    # parser.add_argument('--dim', nargs="+", type=int, default=(256, 256, 256))
    parser.add_argument('--nEpochs', type=int, default=9999)
    parser.add_argument('--classes', type=int, default=21)
    parser.add_argument('--joints', type=int, default=25)
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
    parser.add_argument('--lr', default=1e-4, type=float,  help='learning rate (default: 1e-2)')
    # parser.add_argument('--lr', default=1e-6, type=float,  help='learning rate (default: 1e-2)')
    parser.add_argument('--split', default=0.7, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--worker', default=0, type=int, help="workers")
    
    parser.add_argument('--segonly', action='store_true', default=False)
    parser.add_argument('--segnet', type=str, default="unet3d")
    parser.add_argument('--joint_center_idx', type=int, default=0)
    parser.add_argument('--use_lbs', action='store_true', default=False)
    parser.add_argument('--encoderonly', action='store_true', default=False)

    parser.add_argument('--jointonly', action='store_true', default=False)
    parser.add_argument('--sigma_scale', default=100.0, type=float, help='sigma scale for generating heatmap')
    parser.add_argument('--weight_reg', default=0.0005, type=float)

    parser.add_argument('--loadData', default=True) 
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--model', type=str, default='MRIJOINTNET',
                        choices=('MRIBONENET', 'MRIJOINTNET'))
    parser.add_argument('--model_method', type=str, default='UNET',
                        choices=('SCN', 'UNET'))


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
