import sys
sys.path.append("/p300/liyuwei/MRI_Bonenet/MedicalZooPytorch")
sys.path.append("F:\\OneDrive\\Projects_ongoing\\10_HANDMRI\\mri_bone_net\\MedicalZooPytorch\\")
from torch.functional import norm
import torch
import torch.nn as nn
from lib.losses3D.BaseClass import _AbstractDiceLoss
from lib.losses3D.basic import *
from torch.nn import MSELoss
from medpy import metric
import numpy as np
from lib.medzoo.SCNet_op import generate_heatmap_target

def unravel_indices(indices,shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def batch_argmax(heatmap, batch_dim=1):
    """
    Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
    and returns the indices of the max element of each "batch row".
    More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
    the indices of the max element of tensor[v].
    """
    if batch_dim >= len(heatmap.shape):
        raise NoArgMaxIndices()
    batch_shape = heatmap.shape[:batch_dim]
    non_batch_shape = heatmap.shape[batch_dim:]
    flat_non_batch_size = torch.prod(torch.tensor(non_batch_shape))
    heatmap_with_flat_non_batch_portion = heatmap.reshape(*batch_shape, flat_non_batch_size)

    dimension_of_indices = len(non_batch_shape)

    # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
    # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
    # is empty. We cover that case first.
    if heatmap_with_flat_non_batch_portion.numel() == 0:
        # If empty, either the batch dimensions or the non-batch dimensions are empty
        batch_size = torch.prod(batch_shape)
        if batch_size == 0:  # if batch dimensions are empty
            # return empty tensor of appropriate shape
            batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
        else:  # non-batch dimensions are empty, so argmax indices are undefined
            raise NoArgMaxIndices()
    else:   # We actually have elements to maximize, so we search for them
        indices_of_non_batch_portion = heatmap_with_flat_non_batch_portion.argmax(dim=-1)
        batch_of_unraveled_indices = unravel_indices(indices_of_non_batch_portion, non_batch_shape)

    if dimension_of_indices == 1:
        # above function makes each unraveled index of a n-D tensor a n-long tensor
        # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
        batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
    return batch_of_unraveled_indices


def get_batch_channel_image_size(inputs, data_format="channels_first"):
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        if len(inputs_shape) == 4:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:4]
        if len(inputs_shape) == 5:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:5]
    elif data_format == 'channels_last':
        if len(inputs_shape) == 4:
            return inputs_shape[0], inputs_shape[3], inputs_shape[1:3]
        if len(inputs_shape) == 5:
            return inputs_shape[0], inputs_shape[4], inputs_shape[1:4]

def project_3d_joint(target_landmark, affine):
    target_landmark = target_landmark.view(-1, 3)
    hom = torch.ones(target_landmark.shape[0], 1).type(torch.float).to(target_landmark.device)
    target_landmark_hom = torch.cat([target_landmark, hom], dim=-1)
    target_landmark_idx = torch.matmul(torch.inverse(affine), target_landmark_hom.T).permute(0,2,1)[:,:,:3] 
    return target_landmark_idx


class PayerLoss(nn.Module):

    def __init__(self, joints, heatmap_size, sigma_scale, weight_heatmap=1000, weight_sigma=1000, weight_reg=0.0005):
        super(PayerLoss, self).__init__()
        self.n_joints = joints
        self.l2_loss = MSELoss()
        self.heatmap_size = heatmap_size
        self.sigma_scale = sigma_scale
        self.weight_heatmap = weight_heatmap
        self.weight_sigma = weight_sigma
        self.weight_reg = weight_reg

    def forward(self, pred, target, val=False):
        target_landmark, affine = target
        pred_heatmap, sigmas = pred

        target_landmark_idx = project_3d_joint(target_landmark, affine)
        target_heatmap = self.generate_heatmap(target_landmark_idx, sigmas)
        
        loss_all = self.weight_heatmap * self.loss_heatmap(pred_heatmap, target_heatmap) + self.weight_sigma * self.loss_sigmas(sigmas, target_landmark_idx)

        jerror = self.joint_error(pred_heatmap, target_landmark_idx)
        print("jerror: ", jerror)
        # meta = {"jointerror": jerror}
        return loss_all, None, None

    def generate_heatmap(self, target_landmark_idx, sigmas):
        target_heatmap = generate_heatmap_target(self.heatmap_size, target_landmark_idx, sigmas, self.sigma_scale, normalize=True)
        return target_heatmap

    def joint_error_project(self, pred_heatmap, target_landmark, affine):
        target_landmark_idx = project_3d_joint(target_landmark, affine)
        return self.joint_error(pred_heatmap, target_landmark_idx)

    def joint_error(self, pred_heatmap, target_landmark_idx):
        # argmax
        pred_landmark_idx = batch_argmax(pred_heatmap, batch_dim=2)
        pred_landmark_idx = pred_landmark_idx.view(target_landmark_idx.shape)
        jerr = torch.mean(torch.sqrt(torch.sum((pred_landmark_idx - target_landmark_idx)**2, dim=-1)))
        return jerr

    # def loss_reg(self, net_weight):
    #     l2_regularization = torch.norm(net_weight, 2)
    #     return l2_regularization

    def loss_heatmap(self, prediction, target_heatmap):
        batch_size = target_heatmap.shape[0]
        return self.l2_loss(prediction, target_heatmap)
        # return (torch.sum((target_heatmap-prediction)**2) / 2.0) / batch_size

    def loss_sigmas(self, sigmas, target_landmark):
        # target_landmark : B, 25, 3
        batch_size = target_landmark.shape[0]
        # sum(t ** 2) / 2
        # return (torch.sum((sigmas.unsqueeze(0) * target_landmark[:, :, 0])**2) / 2.0 ) / batch_size
        # return torch.sum((sigmas**2) / 2.0)
        return self.l2_loss(sigmas, torch.zeros_like(sigmas))

if __name__ == "__main__":
    crt = PayerLoss(25)
    pred = torch.rand(2, 15, 12,12,12)
    target = torch.rand(2, 15, 12,12,12)
    loss = crt((pred, torch.rand(15),torch.rand(1)), (target, torch.rand(2, 15, 3)))
    print(loss)