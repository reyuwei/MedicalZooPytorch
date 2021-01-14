# Python libraries
# %%
import argparse
import os

import torch
import numpy as np
import sys
sys.path.append("/p300/liyuwei/MRI_Bonenet/MedicalZooPytorch")
# %%
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
from lib.train.trainer import prepare_input
# Lib files
import lib.utils as utils
from lib.losses3D.JoinLoss import JoinLoss
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 1777777
torch.manual_seed(seed)
import SimpleITK as sitk

# %%
def main():

    args = argparse.ArgumentParser()
    args = args.parse_args()
    parserfile = "/p300/liyuwei/MRI_Bonenet/saved_models/MRIBONENET_checkpoints/MRIBONENET_11_01___19_22_mrihand_args.txt"
    with open(parserfile, 'r') as f:
        args.__dict__ = json.load(f)
    print(args)
    args.resume = args.save

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    training_generator, val_generator, full_volume, affine = \
                    medical_loaders.generate_datasets(args, path='/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final')
                    
    model, optimizer = medzoo.create_model(args)
    criterion = JoinLoss(classes=args.classes, skip_index_after=args.classes)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)

    print("START VAL...")
    out_metric, (joint_output, joint_gt), (seg_output, seg_gt) = trainer.test()
    
    seg_output = torch.argmax(seg_output, dim=1).squeeze()

    os.makedirs(args.save + "/eval", exist_ok=True)
    print(out_metric)

    # with open(args.save + "/eval" + "/metric.json", 'w') as f:
        # json.dump(out_metric, f, indent=2)

    joint_output = joint_output.cpu().detach().numpy()
    joint_gt = joint_gt.cpu().detach().numpy()
    seg_output = seg_output.cpu().detach().numpy()
    seg_gt = seg_gt.cpu().detach().numpy()
    idx = 0
    if len(joint_output)!=0:
        for jo, jg, seg_o, seg_g in zip(joint_output, joint_gt, seg_output, seg_gt):
            idx += 1
            np.save(args.save + "/eval" + "/{:02d}_joint_output.xyz".format(idx), jo)
            np.save(args.save + "/eval" + "/{:02d}_joint_gt.xyz".format(idx), jg)

            sitk.WriteImage(sitk.GetImageFromArray(seg_o), args.save + "/eval" + "/{:02d}_sego.nii".format(idx))
            sitk.WriteImage(sitk.GetImageFromArray(seg_g), args.save + "/eval" + "/{:02d}_segg.nii".format(idx))
    else:
        for seg_o, seg_g in zip(seg_output, seg_gt):
            idx += 1
            sitk.WriteImage(sitk.GetImageFromArray(seg_o), args.save + "/eval" + "/{:02d}_sego.nii".format(idx))
            sitk.WriteImage(sitk.GetImageFromArray(seg_g), args.save + "/eval" + "/{:02d}_segg.nii".format(idx))

        

# %%

if __name__ == '__main__':
    main()

# %%
