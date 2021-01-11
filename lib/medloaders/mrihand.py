import sys
sys.path.append("/p300/liyuwei/MRI_Bonenet/MedicalZooPytorch")
import os

import numpy as np
from torch.utils.data import Dataset

import lib.utils as utils
import lib.augment3D as augment3D
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import find_random_crop_dim
from lib.medloaders.medical_image_process import crop_img
import torch
from pathlib import Path

def mask2xyz(mask, affine):
    grid = create_grid(mask.shape)
    index = grid[mask.reshape(-1)!=0, :]
    pp = Index2PhysicalPoint(index.numpy(), affine)
    np.savetxt("tmp.xyz", pp)

def create_grid(volume_size):
    volume_size = volume_size
    xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size[0]), 
                                    torch.arange(volume_size[1]), 
                                    torch.arange(volume_size[2]))
    grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
    grid = grid.reshape((-1, 3))
    return grid

def Index2PhysicalPoint(index, affine):
    if isinstance(index, torch.Tensor):
        affine_4 = torch.eye(4).type_as(affine).to(affine.device)
        affine_4[:3, :4] = affine[:3, :4]
        index_hom = torch.ones(index.shape[0], 4).type_as(index).to(index.device)
        index_hom[:, :3] = index[:, :3]
        point = affine_4 @ index_hom.T    
        return point.T[:, :3]
    else:
        affine_4 = np.eye(4)
        affine_4[:3, :4] = affine[:3, :4]
        index_hom = np.ones([index.shape[0], 4])
        index_hom[:, :3] = index[:, :3]
        point = affine_4 @ index_hom.T    
        return point.T[:, :3]

def PhysicalPoint2Index(point, affine):
    if isinstance(point, torch.Tensor):
        affine_4 = torch.eye(4).type_as(affine).to(affine.device)
        affine_4[:3, :4] = affine[:3, :4]
        point_hom = torch.ones(point.shape[0], 4).type_as(point).to(point.device)
        point_hom[:, :3] = point[:, :3]
        index = affine_4 @ point_hom.T    
        return index.T[:, :3]
    else:
        affine_4 = np.eye(4)
        affine_4[:3, :4] = affine[:3, :4]
        point_hom = np.ones([point.shape[0], 4])
        point_hom[:, :3] = point[:, :3]
        index = np.linalg.inv(affine_4) @ point_hom.T
        return index.T[:, :3]

def crop_pad(t1, s, affine, crop_size):
    full_vol_dim = t1.shape
    
    # pad zero
    pad_0 = (0,0)
    pad_1 = (0,0)
    pad_2 = (0,0)
    if full_vol_dim[0] < crop_size[0]:
        gap = np.ceil(crop_size[0] - full_vol_dim[0] / 2)
        pad_0 =  (int(gap), int(gap))
    if full_vol_dim[1] < crop_size[1]:
        gap = np.ceil((crop_size[1] - full_vol_dim[1]) / 2)
        pad_1 =  (int(gap), int(gap))
    if full_vol_dim[2] < crop_size[2]:
        gap = np.ceil((crop_size[2] - full_vol_dim[2]) / 2)
        pad_2 = (int(gap), int(gap))


    t1 = np.pad(t1, (pad_0, pad_1, pad_2), 'constant')
    s = np.pad(s, (pad_0, pad_1, pad_2), 'constant')

    pad_mat = np.eye(4)
    pad_mat[:3, -1] = np.array([-pad_0[0], -pad_1[0], -pad_2[0]])
    affine = affine @ pad_mat

    crop_start = find_random_crop_dim(t1.shape, crop_size) 
    crop_start_fix = np.array(crop_start)
    for i in range(len(crop_start)):
        if crop_start[i] == crop_size[i]:
            crop_start_fix[i] = 0
    crop_start_fix = tuple(crop_start_fix)
    t1, crop_mat = crop_img(t1, crop_size, crop_start_fix)
    s, _ = crop_img(s, crop_size, crop_start_fix)

    affine = affine @ crop_mat

    return t1, s, affine




class MRIHandDataset(Dataset):
    # "/p300/liyuwei/DATA_mri/Hand_MRI_segdata/nnUNet_preprocessed/Task1074_finegrained_bone_real_90"
    def __init__(self, args, mode, dataset_path='./datasets', crop=False, crop_dim=(200, 200, 200), 
                 lst=None, load=False, voxel_spacing=[0.5, 0.5, 0.5]):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.CLASSES = 21
        self.split_lst = lst
        self.voxel_spacing = voxel_spacing

        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.crop_size = crop_dim
        self.bbx_scale = 1.25
        self.full_volume = None

        # subvol_spacing = str(self.voxel_spacing[0]) + 'x' + str(self.voxel_spacing[1]) + 'x' + str(self.voxel_spacing[2])
        subvol = '_vol_' + str(self.voxel_spacing[0]) + 'x' + str(self.voxel_spacing[1]) + 'x' + str(self.voxel_spacing[2])
        self.sub_vol_path = self.root + '/generated/' + self.mode + subvol + '/'
        self.save_name = self.sub_vol_path + mode + '.pkl'
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
                img_t1_tensor, affine, orig_spacing = img_loader.load_medical_image_hand(self.list_t1[i], type="T1", resample=self.voxel_spacing,
                                                            normalization=self.normalization)
                label_tensor, affine, _ = img_loader.load_medical_image_hand(self.labels[i], type="label", resample=self.voxel_spacing, 
                                                            orig_spacing=orig_spacing)
                np.save(f_t1, img_t1_tensor)
                np.save(f_t1_affine, affine)
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
        if not os.path.exists(item['input']):
            return
        t1 = np.load(item['input'])
        s = np.load(item['target_mask'])
        joint = item['joint']
        affine = item["affine"]
        if t1.shape != s.shape:
            print(item['input'])

        t1, s, affine = crop_pad(t1, s, affine, self.crop_size)
        # print(t1.shape)
        # print(s.shape)

        joint_tensor = torch.from_numpy(joint).float()

        if self.mode == "train" and self.augmentation:
            [augmented_t1], augmented_s, augmented_affine = self.transform([t1], s, affine)
            return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy()), \
                    torch.from_numpy(augmented_affine).float(), joint_tensor

        affine_tensor = torch.from_numpy(affine).float()
        return torch.FloatTensor(t1).unsqueeze(0), torch.FloatTensor(s), affine_tensor, joint_tensor


if __name__ == "__main__":
    path = '/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final'
    total_data = 90
    split_pkl = os.path.join(path, "splits_final.pkl")
    split = np.load(split_pkl, allow_pickle=True)[0]
    train_lst = split['train']
    val_lst = split['val']

    class tmp():
        def __init__(self):
            self.threshold = 0.1
            self.normalization = "mean"
            self.augmentation = True
    
    args = tmp()
    train_loader = MRIHandDataset(args, 'train', dataset_path=path, crop_dim=(128, 128, 128),
                                    lst=train_lst, load=True)
    val_loader = MRIHandDataset(args, 'val', dataset_path=path, crop_dim=(32, 32, 32), 
                                    lst=val_lst, load=True)

    for i in range(len(train_loader)):
        print(3)
        t1, tar = train_loader.__getitem__(3)
        print(t1.shape)
        print(tar.shape)

    # for i in range(len(val_loader)):
    #     print(i)
    #     val_loader.__getitem__(i)
    
    # import SimpleITK as sitk
    # sitk.WriteImage(sitk.GetImageFromArray(tt1.squeeze().numpy()), "tmp_tr.nii")
    # sitk.WriteImage(sitk.GetImageFromArray(ts.numpy()), "tmp_tr_t.nii")

    # tt1, ts = train_loader.__getitem__(1)
    # sitk.WriteImage(sitk.GetImageFromArray(tt1.squeeze().numpy()), "tmp_tr2.nii")
    # sitk.WriteImage(sitk.GetImageFromArray(ts.numpy()), "tmp_tr_t2.nii")
    
    # ttt1, tts = val_loader.__getitem__(1)
    # sitk.WriteImage(sitk.GetImageFromArray(ttt1.squeeze().numpy()), "tmp_ts.nii")
    # sitk.WriteImage(sitk.GetImageFromArray(tts.numpy()), "tmp_ts_t.nii")