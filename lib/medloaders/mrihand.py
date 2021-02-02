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
import scipy.ndimage as ndimage

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
        gap = np.ceil((crop_size[0] - full_vol_dim[0]) / 2)
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


def scale_pad(t1, crop_size):
    full_vol_dim = t1.shape
    # pad zero
    pad_0 = (0,0)
    pad_1 = (0,0)
    pad_2 = (0,0)
    pad_goal = np.max([np.max(full_vol_dim), np.max(crop_size)])
    cube_pad_size = [pad_goal, pad_goal, pad_goal]
    if full_vol_dim[0] < cube_pad_size[0]:
        gap = np.ceil((cube_pad_size[0] - full_vol_dim[0]) / 2)
        pad_0 =  (int(gap), int(gap))
    if full_vol_dim[1] < cube_pad_size[1]:
        gap = np.ceil((cube_pad_size[1] - full_vol_dim[1]) / 2)
        pad_1 =  (int(gap), int(gap))
    if full_vol_dim[2] < cube_pad_size[2]:
        gap = np.ceil((cube_pad_size[2] - full_vol_dim[2]) / 2)
        pad_2 = (int(gap), int(gap))

    t1 = np.pad(t1, (pad_0, pad_1, pad_2), 'constant')
    # s = np.pad(s, (pad_0, pad_1, pad_2), 'constant')

    depth, height, width = t1.shape
    scale = [crop_size[0] * 1.0 / depth, crop_size[1] * 1.0 / height, crop_size[2] * 1.0 / width]
    voxel_size = [0.5, 0.5, 0.5]
    print(np.array(voxel_size) / np.array(scale))
    scaled = ndimage.interpolation.zoom(t1, scale, order=3)
    # scaled_s = ndimage.interpolation.zoom(s, scale, order=0)
    # return scaled, scaled_s
    return scaled, None


def even_pad(t1, s, affine, crop_size):
    full_vol_dim = t1.shape
    cov = 32
    # pad zero
    pad_0 = (0,0)
    pad_1 = (0,0)
    pad_2 = (0,0)
    cube_pad_size = np.array(full_vol_dim)
    for i in range(len(cube_pad_size)):
        if cube_pad_size[i] % cov!=0:
            cube_pad_size[i] = ((full_vol_dim[i] // cov) + 1) * cov
            
    # print(cube_pad_size)

    if full_vol_dim[0] < cube_pad_size[0]:
        gap = np.ceil((cube_pad_size[0] - full_vol_dim[0]) / 2)
        pad_0 =  (int(gap), int(gap))
    if full_vol_dim[1] < cube_pad_size[1]:
        gap = np.ceil((cube_pad_size[1] - full_vol_dim[1]) / 2)
        pad_1 =  (int(gap), int(gap))
    if full_vol_dim[2] < cube_pad_size[2]:
        gap = np.ceil((cube_pad_size[2] - full_vol_dim[2]) / 2)
        pad_2 = (int(gap), int(gap))

    t1 = np.pad(t1, (pad_0, pad_1, pad_2), 'constant')
    s = np.pad(s, (pad_0, pad_1, pad_2), 'constant')

    crop_size = cube_pad_size
    crop_start_fix = [0,0,0] 
    t1, crop_mat = crop_img(t1, crop_size, crop_start_fix)
    s, _ = crop_img(s, crop_size, crop_start_fix)
    # print(t1.shape)
    # print(s.shape)

    return t1, s


class MRIHandDataset(Dataset):
    # "/p300/liyuwei/DATA_mri/Hand_MRI_segdata/nnUNet_preprocessed/Task1074_finegrained_bone_real_90"
    def __init__(self, args, mode, dataset_path='./datasets', crop_dim=(200, 200, 200), 
                 lst=None, load=False, voxel_spacing=[0.5, 0.5, 0.5], seg_only=False):
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
        self.seg_only = seg_only

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
        
        self.gt_param = None
        if not seg_only:
            if os.path.exists(os.path.join(self.root, "gt_param_dict.pkl")):
                self.gt_param = np.load(os.path.join(self.root, "gt_param_dict.pkl"), allow_pickle=True)
        
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



        if "t1" in self.root:
            for x in self.split_lst:
                self.list_t1.append(os.path.join(self.root, x + "_t1.nii"))
                self.labels.append(os.path.join(self.root,  x + "_finegrained_bone_49D.nii"))
                self.list_joint.append(os.path.join(self.root, x + "_joints_3d.txt"))
                joint_idx = np.loadtxt(os.path.join(self.root, x + "_joints_idx.txt")).reshape(-1, 3)
                self.joint_bbx.append(self.create_bbx_from_joint(joint_idx, scale=self.bbx_scale))
        else:
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
            if "t1" in self.root:
                name = Path(self.list_t1[i]).stem[:-3]
            else:
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
        input_tensor, target, affine_mat, joint, input_scale, params_gt = self.fetch(index)
        if "theta" in params_gt:
            theta = params_gt['theta']
            beta = params_gt['beta']
            trans = params_gt['trans']
        else:
            theta = torch.FloatTensor()
            beta = torch.FloatTensor()
            trans = torch.FloatTensor()
        return input_tensor, target, affine_mat, joint, input_scale, theta, beta, trans


    def fetch(self, index):
        item = self.data_dict[index]
        # print(item['input'])
        if not os.path.exists(item['input']):
            return
        
        param_dict = {}

        if self.gt_param is not None:
            name = Path(item['input']).stem[:-3]
            if name in self.gt_param:
                params = self.gt_param[name]
                theta = torch.from_numpy(params['theta']).float()
                beta = torch.from_numpy(params['beta']).float()
                trans_back2gt = torch.from_numpy(params['trans_back2gt']).float()
                param_dict = {
                    'theta': theta,
                    'beta': beta,
                    'trans': trans_back2gt
                }


        t1 = np.load(item['input'])
        s = np.load(item['target_mask'])
        joint = item['joint']
        affine = item["affine"]
        if t1.shape != s.shape:
            print(item['input'])

        joint_tensor = torch.from_numpy(joint).float()
        
        if self.seg_only:
            t1, s, affine = crop_pad(t1, s, affine, self.crop_size)
            # t1, s = scale_pad(t1, s, affine, self.crop_size)
            # t1, s = even_pad(t1, s, affine, self.crop_size)

            if self.mode == "train" and self.augmentation:
                [augmented_t1], augmented_s, augmented_affine = self.transform([t1], s, affine)
                return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy()), \
                        torch.from_numpy(augmented_affine).float(), joint_tensor, torch.FloatTensor(), param_dict
            else:
                affine_tensor = torch.from_numpy(affine).float()
                return torch.FloatTensor(t1).unsqueeze(0), torch.FloatTensor(s), affine_tensor, joint_tensor, torch.FloatTensor(), param_dict
        
        else:
            if self.mode == "train" and self.augmentation:
                [augmented_t1], augmented_s, augmented_affine = self.transform([t1], s, affine)
                
                augmented_t1_scale, _ = scale_pad(augmented_t1, self.crop_size)
                augmented_t1_crop, augmented_s_crop, augmented_affine_crop = crop_pad(augmented_t1, augmented_s, augmented_affine, self.crop_size)
                
                return torch.FloatTensor(augmented_t1_crop.copy()).unsqueeze(0), torch.FloatTensor(augmented_s_crop.copy()), \
                        torch.from_numpy(augmented_affine_crop).float(), joint_tensor,  torch.FloatTensor(augmented_t1_scale.copy()).unsqueeze(0), param_dict
            else:
                scaled_t1, _ = scale_pad(t1, self.crop_size)
                t1_crop, s_crop, affine_crop = crop_pad(t1, s, affine, self.crop_size)
                affine_tensor = torch.from_numpy(affine_crop).float()
                return torch.FloatTensor(t1_crop).unsqueeze(0), torch.FloatTensor(s_crop), affine_tensor, \
                        joint_tensor, torch.FloatTensor(scaled_t1).unsqueeze(0), param_dict





if __name__ == "__main__":
    path = '/p300/liyuwei/DATA_mri/Hand_MRI_capture/seg_final_t1'
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
                                    lst=train_lst, load=True, seg_only=False)
    val_loader = MRIHandDataset(args, 'val', dataset_path=path, crop_dim=(32, 32, 32), 
                                    lst=val_lst, load=True, seg_only=False)

    for i in range(len(train_loader)):
        print(3)
        t1, tar, affine, joint, scaled_t1, param_dict = train_loader.__getitem__(3)
        print(param_dict)


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