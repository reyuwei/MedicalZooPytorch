import glob
from lib.augment3D.random_shift import RandomShift
import os

import numpy as np
from torch.utils.data import Dataset

import lib.utils as utils
import lib.augment3D as augment3D
from lib.medloaders import medical_image_process as img_loader, nnunet_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes
from lib.medloaders.medical_loader_utils import get_viz_set
import torch
from pathlib import Path


class MRIHandDataset(Dataset):
    # "/p300/liyuwei/DATA_mri/Hand_MRI_segdata/nnUNet_preprocessed/Task1074_finegrained_bone_real_90"
    def __init__(self, args, mode, dataset_path='./datasets', crop=False, crop_dim=(200, 200, 200), 
                 lst=None, samples=1000, load=False, voxel_spacing=[0.5, 0.5, 0.5]):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.CLASSES = 20
        self.split_lst = lst
        self.voxel_spacing = voxel_spacing

        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.crop_size = crop_dim
        self.bbx_scale = 1.25
        self.samples = samples
        self.full_volume = None

        subvol_spacing = str(self.voxel_spacing[0]) + 'x' + str(self.voxel_spacing[1]) + 'x' + str(self.voxel_spacing[2])
        subvol = '_vol_' + str(self.crop_size[0]) + 'x' + str(self.crop_size[1]) + 'x' + str(self.crop_size[2])
        self.sub_vol_path = self.root + '/generated/' + self.mode + subvol + "-" + subvol_spacing + '/'
        self.save_name = self.sub_vol_path + mode + '-samples-' + str(samples) + '.pkl'
        os.makedirs(self.sub_vol_path, exist_ok=True)

        # utils.make_dirs(self.sub_vol_path)

        self.data_dict = []

        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), 
                            augment3D.RandomFlip(),
                            augment3D.RandomShift(),
                            augment3D.RandomRotation()], p=0.5)

        if load and os.path.exists(self.save_name):
            ## load pre-generated data
            self.data_dict = utils.load_list(self.save_name)
            self.affine = self.data_dict[0]['affine']
            self.full_volume = None
            return

        self.list_t1 = []
        self.labels = []
        self.list_joint = []
        self.joint_bbx = []
        for x in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, x)):
                if self.split_lst is not None:
                    if x in self.split_lst:
                        self.list_t1.append(os.path.join(self.root, x, x + "_t1.nii"))
                        self.labels.append(os.path.join(self.root, x, x + "_finegrained_bone.nii"))
                        self.list_joint.append(os.path.join(self.root, x, x + "_joints_3d.txt"))
                        joint_idx = np.loadtxt(os.path.join(self.root, x, x + "_joints_idx.txt")).reshape(-1, 3)
                        self.joint_bbx.append(self.create_bbx_from_joint(joint_idx, scale=self.bbx_scale))
        
        self.create_input_data()
        
        self.affine = self.data_dict[0]['affine']
        self.full_volume = None

        utils.save_list(self.save_name, self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def create_input_data(self):
        total = len(self.list_t1)        
        print(self.mode, "Dataset samples :", total)

        for i in range(total):
            name = Path(self.list_t1[i]).parent.stem
            print(name)
            f_t1 = self.sub_vol_path + name + '_t1.npy'
            f_t1_mask = self.sub_vol_path + name + '_t1_gt.npy'
            f_t1_affine = self.sub_vol_path + name + '_affine.npy'
            joint = np.loadtxt(self.list_joint[i]).reshape(-1, 3)

            bbx = self.joint_bbx[i]
            start = np.floor(bbx['topleft'])
            bbx_size = np.floor(bbx['size'])
            
            if os.path.exists(f_t1_affine):
                affine = np.load(f_t1_affine)

            if not os.path.exists(f_t1):
                img_t1_tensor, affine = img_loader.load_medical_image_hand(self.list_t1[i], type="T1", resample=self.voxel_spacing,
                                                            normalization=self.normalization, rescale=self.crop_size)
                np.save(f_t1, img_t1_tensor)
                np.save(f_t1_affine, affine)

            if not os.path.exists(f_t1_mask):
                label_tensor, affine = img_loader.load_medical_image_hand(self.labels[i], type="label", 
                                                            resample=self.voxel_spacing, rescale=self.crop_size)
                np.save(f_t1_mask, label_tensor)
        
            self.data_dict.append({
                "input": f_t1, 
                "target_mask": f_t1_mask,
                "joint": joint,
                "affine": affine
            })

    
    def create_bbx_from_joint(self, joint_idx, scale=1.15):
        joint_bbx_min = np.floor(np.min(joint_idx, axis=0))
        joint_bbx_max = np.ceil(np.max(joint_idx, axis=0))
        bbx_center = (joint_bbx_min + joint_bbx_max) / 2
        bbx_size = scale * (joint_bbx_max - joint_bbx_min)
        bbx_topleft = bbx_center - bbx_size / 2
        bbx_topleft = np.max(bbx_topleft, 0)
        bbx = {"center": bbx_center, "size": bbx_size, "topleft": bbx_topleft}
        return bbx

    def __getitem__(self, index):
        item = self.data_dict[index]
        t1 = np.load(item['input'])
        s = np.load(item['target_mask'])
        joint = item['joint']
        affine = item["affine"]

        s_expand = np.zeros([self.CLASSES, s.shape[0], s.shape[1], s.shape[2]])
        for i in range(0, self.CLASSES):
            s_expand[i][s==i+1] = 1
        # s[s>self.CLASSES] = 0
        # s = s - 1

        if self.mode == "train" and self.augmentation:
            [augmented_t1], augmented_s = self.transform([t1], s_expand)
            return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy())

        return torch.FloatTensor(t1).unsqueeze(0), torch.FloatTensor(s_expand)