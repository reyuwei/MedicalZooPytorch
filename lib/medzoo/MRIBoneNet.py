from lib.losses3D.basic import expand_as_one_hot
import sys
sys.path.append("/p300/liyuwei/MRI_Bonenet/MedicalZooPytorch")
import os
import pxr
import kaolin
from types import new_class
from lib.medzoo.ResNet3D_VAE import ResNetEncoder
import torch.nn as nn
import torch
from torchsummary import summary
import torchsummaryX
from lib.medzoo import Densenet3D
from lib.medzoo.BaseModelClass import BaseModel
from lib.medzoo.Unet3D import UNet3D
from lib.medzoo.Vnet import VNet
from lib.medzoo.ResNet3D_VAE import ResNet3dVAE
from lib.medzoo.COVIDNet import Flatten
from lib.bonepth.bonelayer import BoneLayer
import lib.bonepth.globalvar as globalvar
import numpy as np
# import pytorch3d

class MRIBoneNet(BaseModel):
    def __init__(self, in_channels, classes=21, base_n_filter=8, seg_only=False, 
                    center_idx=8, use_lbs=True, seg_net="unet3d", encoder_only=False):
        super(MRIBoneNet, self).__init__()

        self.seg_only = seg_only
        self.in_channels = in_channels
        self.n_classes = classes
        self.base_n_filter=base_n_filter

        self.pose_param_dim = globalvar.STATIC_BONE_NUM*3#?
        self.shape_param_dim = 35#?

        self.use_lbs = use_lbs
        if not self.seg_only:
            self.encoder = ResNetEncoder(in_channels=self.in_channels, start_channels=32)
            self.num_latent_code = self.encoder.out_channel
            self.flatten = Flatten()

            self.fc_pose = nn.Sequential(
                nn.Linear(self.num_latent_code*16*16*16, 512),
                nn.ReLU(),
                nn.Linear(512, self.pose_param_dim),
                # nn.Tanh()
            )
            self.fc_shape = nn.Sequential(
                nn.Linear(self.num_latent_code*16*16*16, 512),
                nn.ReLU(),
                nn.Linear(512, self.shape_param_dim)
            )

            self.center_idx = center_idx
            self.use_jreg = not self.use_lbs
            self.use_shapepca = not self.use_lbs
            template_dict = np.load("lib/bonepth/bone_template.pkl", allow_pickle=True)
            # joint_regressor = np.load("lib/bonepth/J_regressor.pkl", allow_pickle=True)
            self.bonelayer = BoneLayer(template_dict, center_idx=self.center_idx, ncomps_shape=self.shape_param_dim, use_jreg=self.use_jreg, use_shapepca=self.use_shapepca)
            self.seg_in_channels = 2    
        else:
            self.seg_in_channels = 1

        self.encoder_only = encoder_only
        if not self.encoder_only:
            if seg_net == "unet3d":
                self.segnet = UNet3D(in_channels=self.seg_in_channels, n_classes=self.n_classes, base_n_filter=self.base_n_filter)
            elif seg_net == "vnet":
                self.segnet = VNet(in_channels=self.seg_in_channels, classes=self.n_classes)
            else:
                assert "no such segnet!!"


    def map(self, x, affine, verts):
        # porject mesh to mask
        
        # create grid
        volume_size = x.shape[-3:]
        device = x.device
        xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size[0], device=device), 
                                        torch.arange(volume_size[1], device=device), 
                                        torch.arange(volume_size[2], device=device))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.reshape((-1, 3))
        hom = torch.ones(grid.shape[0], 1).type(torch.float).to(device)
        grid_hom = torch.cat([grid, hom], dim=-1)
        points = torch.matmul(affine, grid_hom.T).permute(0, 2, 1)[:,:, :3]  # N, volume_size**3, 3
        
        # assign label with pytorch3d
        sep_vert, spe_face = self.bonelayer.verts_separater, self.bonelayer.faces_separater
        counter_v = 0
        counter_f = 0
        id = 0
        mask_flat = torch.zeros_like(x).to(device).view(x.shape[0], -1)
        for v, f in zip(sep_vert, spe_face):
            if counter_v == v:
                continue
            id += 1
            # print(counter_v, v, counter_f, f)
            bone_verts = verts[:, counter_v:v, :]
            bone_faces = self.bonelayer.th_faces[counter_f:f, :] - counter_v
            sign = kaolin.ops.mesh.check_sign(bone_verts, bone_faces, points) # B, 128**3
            mask_flat[sign==True] = id
            counter_v = v
            counter_f = f

        mask = mask_flat.view(x.shape)
        return mask

    def forward_bone(self, x, x_full, affine, root, trans_mat):
        '''
        x: [B, 1, H, W, D]
        feature: [B, FC, 32, 32, 32]
        pose: [B, pose_dim]
        shape: [B, shape_dim]
        proj_mask: [B, C, H, W, D]
        '''
        batch_size = x.shape[0]
        _, _, _, x4 = self.encoder(x_full)
        flattened = self.flatten(x4)
        
        pose = self.fc_pose(flattened).view(batch_size,-1, 3)
        shape = self.fc_shape(flattened).view(batch_size, -1)

        if self.use_lbs: # pose and scale
            scale = torch.tanh(shape) + torch.ones_like(shape)
            verts_con, joints_con = self.bonelayer(pose, th_scale=scale)
        else: # pose and shape
            verts_con, joints_con = self.bonelayer(pose, th_shape_param=shape)
        
        hom = torch.ones(verts_con.shape[0], verts_con.shape[1], 1).type(torch.float).to(verts_con.device)
        verts_hom = torch.cat([verts_con, hom], dim=-1).permute(0, 2, 1)
        verts = torch.bmm(trans_mat, verts_hom).permute(0, 2, 1)[:, :, :3]  # N, volume_size**3, 3

        homj = torch.ones(joints_con.shape[0],joints_con.shape[1], 1).type(torch.float).to(joints_con.device)
        joints_hom = torch.cat([joints_con, homj], dim=-1).permute(0, 2, 1)
        joints = torch.bmm(trans_mat, joints_hom).permute(0, 2, 1)[:, :, :3]  # N, volume_size**3, 3

        proj_mask = self.map(x, affine, verts)

        if self.encoder_only:
            x_cat = torch.cat([x, proj_mask], dim=1)
            out = self.segnet(x_cat)
        else:
            out = expand_as_one_hot(proj_mask.squeeze(), self.n_classes)

        if self.use_lbs:
            return out, joints, pose, scale
        else:
            return out, joints, pose, shape


    def forward(self, x):
        input_x, affine, joint, input_x_full, trans = x
        if self.seg_only:
            out = self.segnet(input_x)
            return out, None, None, None
        else:
            root = joint[:, self.center_idx, :]
            return self.forward_bone(input_x, input_x_full, affine, root, trans)

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, 1, 128, 128, 128)
        ideal_out = torch.rand(1, self.n_classes, 128, 128, 128)
        out, _, _, _ = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (1, 128, 128, 128),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("MRIBONENET test is complete")

if __name__ == "__main__":
    net = MRIBoneNet(1, classes=21, seg_only=False)

    input_tensor = torch.rand(1, 1, 128, 128, 128)
    affine = torch.rand(1, 4, 4)
    joint = torch.rand(1, 25, 3)
    ideal_out = torch.rand(1, 21, 128, 128, 128)

    out, joint, pose, shape = net.forward((input_tensor, affine))
    print(out.shape)
    print(joint.shape)
    print(pose.shape)
    print(shape.shape)