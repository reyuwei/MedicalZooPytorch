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
from lib.losses3D.JoinLoss import joint_error, mask_error
# %%
def main():

    args = argparse.ArgumentParser()
    args = args.parse_args()
    # parserfile = "/p300/liyuwei/MRI_Bonenet/saved_models/MRIBONENET_checkpoints/MRIBONENET_11_01___19_22_mrihand_args.txt"
    parserfile = "/p300/liyuwei/MRI_Bonenet/saved_models/MRIBONENET_checkpoints/MRIBONENET_14_01___06_07_mrihand_args.txt" #unet
    # parserfile = "/p300/liyuwei/MRI_Bonenet/saved_models/MRIBONENET_checkpoints/MRIBONENET_14_01___06_10_mrihand_args.txt" #vnet
    with open(parserfile, 'r') as f:
        args.__dict__ = json.load(f)
    print(args)
    args.resume = args.save
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    args.dim = (256, 256, 256)
    args.worker =0
    args.batchSz = 1
    args.encoderonly = False
    args.shuffle=False
    params = {'batch_size': 1,
        'shuffle': False,
        'num_workers': 1}

    training_generator, val_generator, full_volume, affine = \
                    medical_loaders.generate_datasets(args, path='/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final_t1', params=params)
                    
    model, optimizer = medzoo.create_model(args)
    criterion = JoinLoss(classes=args.classes, skip_index_after=args.classes)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)

    print("START VAL...")
    # out_metric, (joint_output, joint_gt), (seg_output, seg_gt) , input= trainer.test()
    # seg_output = torch.argmax(seg_output, dim=1).squeeze()
    os.makedirs(args.save + "/eval", exist_ok=True)
    # print(out_metric)
    # with open(args.save + "/eval" + "/metric.json", 'w') as f:
        # json.dump(out_metric, f, indent=2)
    meta = {}



    trainer.model.eval()
    normalization = torch.nn.Sigmoid()
    for batch_idx, input_tuple in enumerate(trainer.valid_data_loader):
        with torch.no_grad():
            print(batch_idx, len(trainer.valid_data_loader))
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=trainer.args)
            output = trainer.model(input_tensor)
            
            x, affine_mat, joint, input_scale, trans = input_tensor
            x = x.squeeze().detach().cpu().numpy()

            pred_mask, pred_joint, pose_param, shape_param = output
            target_mask, target_joint, target_theta, target_beta = target
            normalize_seg_output = normalization(pred_mask)

            mask_meta = mask_error(normalize_seg_output, target_mask)
            meta.update(mask_meta)

            target_mask = target_mask.detach().cpu()

            normalize_seg_output = torch.argmax(normalize_seg_output, dim=1).squeeze().detach().cpu()
            normalize_seg_output = np.array(normalize_seg_output.squeeze(), dtype=int)
            sitk.WriteImage(sitk.GetImageFromArray(normalize_seg_output), args.save + "/eval" + "/{:02d}_sego.nii".format(batch_idx))
            sitk.WriteImage(sitk.GetImageFromArray(target_mask), args.save + "/eval" + "/{:02d}_segg.nii".format(batch_idx))
            sitk.WriteImage(sitk.GetImageFromArray(x), args.save + "/eval" + "/{:02d}_input.nii".format(batch_idx))



    print(meta)
    with open(args.save + "/eval" + "/metric.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # joint_output = joint_output.cpu().detach().numpy()
    # joint_gt = joint_gt.cpu().detach().numpy()
    # seg_output = seg_output.cpu().detach().numpy()
    # seg_output = np.asarray(seg_output, dtype=int)
    # seg_gt = seg_gt.cpu().detach().numpy()
    # input = input.cpu().detach().numpy()
    # idx = 0
    # if len(joint_output)!=0:
    #     for jo, jg, seg_o, seg_g in zip(joint_output, joint_gt, seg_output, seg_gt):
    #         idx += 1
    #         np.save(args.save + "/eval" + "/{:02d}_joint_output.xyz".format(idx), jo)
    #         np.save(args.save + "/eval" + "/{:02d}_joint_gt.xyz".format(idx), jg)

    #         sitk.WriteImage(sitk.GetImageFromArray(seg_o), args.save + "/eval" + "/{:02d}_sego.nii".format(idx))
    #         sitk.WriteImage(sitk.GetImageFromArray(seg_g), args.save + "/eval" + "/{:02d}_segg.nii".format(idx))
    # else:
    #     for seg_o, seg_g, inp in zip(seg_output, seg_gt, input):
    #         idx += 1
    #         sitk.WriteImage(sitk.GetImageFromArray(seg_o), args.save + "/eval" + "/{:02d}_sego.nii".format(idx))
    #         sitk.WriteImage(sitk.GetImageFromArray(seg_g), args.save + "/eval" + "/{:02d}_segg.nii".format(idx))
    #         sitk.WriteImage(sitk.GetImageFromArray(inp), args.save + "/eval" + "/{:02d}_input.nii".format(idx))

        

# %%

if __name__ == '__main__':
    main()

# %%
