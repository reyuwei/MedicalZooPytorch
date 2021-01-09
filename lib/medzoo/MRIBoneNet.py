from lib.medzoo.ResNet3D_VAE import ResNetEncoder
import torch.nn as nn
import torch
from torchsummary import summary
import torchsummaryX
from lib.medzoo.BaseModelClass import BaseModel
from lib.medzoo.Unet3D import UNet3D
from lib.medzoo.COVIDNet import Flatten
from lib.bonepth.bonelayer import BoneLayer


class MRIBoneNet(BaseModel):
    def __init__(self, in_channels, classes=20, base_n_filter=8, seg_only=False):
        super(MRIBoneNet, self).__init__()

        self.seg_only = seg_only
        self.in_channels = in_channels
        self.n_classes = classes
        self.base_n_filter=base_n_filter

        self.pose_param_dim = 20*3
        self.shape_param_dim = 10

        self.num_latent_code = 64

        if not self.seg_only:
            self.encoder = ResNetEncoder(in_channels=self.in_channels, start_channels=32)
            self.segnet = UNet3D(in_channels=2, n_classes=self.n_classes, base_n_filter=base_n_filter )
            self.flatten = Flatten()
            self.fc_pose = nn.Sequential(
                nn.Linear(self.num_latent_code*32*32*32, 512),
                nn.ReLU(),
                nn.Linear(512, self.pose_param_dim),
                nn.Tanh()
            )
            self.fc_shape = nn.Sequential(
                nn.Linear(self.num_latent_code*32*32*32, 512),
                nn.ReLU(),
                nn.Linear(512, self.shape_param_dim),
            )
            self.bonelayer = BoneLayer()
        else:
            self.segnet = UNet3D(in_channels=1, n_classes=self.n_classes, base_n_filter=base_n_filter)

    def map(verts, x_volume):
        # porject mesh to mask
        return mask

    def forward_bone(self, x):
        '''
        x: [B, 1, H, W, D]
        feature: [B, FC, 32, 32, 32]
        pose: [B, pose_dim]
        shape: [B, shape_dim]
        proj_mask: [B, C, H, W, D]
        '''
        x1, x2, x3, x4 = self.encoder(x)
        flattened = self.flatten(x4)
        
        pose = self.fc_pose(flattened)
        shape = self.fc_shape(flattened)

        verts, joints = self.bonelayer(pose, shape)
        proj_mask = self.map(verts, x)

        x_cat = torch.cat([x, proj_mask], dim=1)
        out = self.segnet(x_cat)

        return out, joints, pose, shape


    def forward(self, x):
        if self.seg_only:
            out = self.segnet(x)
            return out, None, None, None
        else:
            self.forward_bone()

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, 1, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out, _, _, _ = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (1, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("MRIBONENET test is complete")