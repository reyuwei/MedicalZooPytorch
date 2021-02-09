from lib.losses3D.basic import expand_as_one_hot
import sys
sys.path.append("/p300/liyuwei/MRI_Bonenet/MedicalZooPytorch")
import os
from types import new_class
import torch.nn as nn
import torch
from torchsummary import summary
import torchsummaryX
import numpy as np
from lib.medzoo.SCNet import Payer_Heatmap_UNet3D, Payer_Heatmap_SCNet
from lib.medzoo.BaseModelClass import BaseModel


class MRIJointNet(BaseModel):
    def __init__(self, in_channels, n_heatmaps=25, method="UNET", num_filters_base=64):
        super(MRIJointNet, self).__init__()

        self.method = method
        self.in_channels = in_channels
        self.n_heatmaps = n_heatmaps
        self.num_filters_base = num_filters_base

        if method == "UNET":
            self.net = Payer_Heatmap_UNet3D(self.in_channels, self.n_heatmaps, num_filters_base=self.num_filters_base)
        elif method == "SCN":
            self.net = Payer_Heatmap_SCNet(self.in_channels, self.n_heatmaps, num_filters_base=self.num_filters_base)
        

    def forward(self, x):
        return self.net(x)

    def test(self):
        pass
