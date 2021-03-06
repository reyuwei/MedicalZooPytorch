from torch.functional import norm
from lib.losses3D.BCE_dice import BCEDiceLoss
from lib.losses3D.BaseClass import _AbstractDiceLoss
from lib.losses3D.basic import *
from torch.nn import MSELoss
from medpy import metric
import numpy as np

def joint_error(pred, gt):
    meta = {}
    j_err = torch.sqrt(torch.sum((gt- pred)**2, axis=1))
    meta['joint_mse'] = torch.mean(j_err).numpy()

    thresholds = torch.linspace(0, 50, 10)
    norm_factor = torch.trapz(torch.ones_like(thresholds), thresholds)

    pck_curve = []
    j_err = j_err.view(-1)
    for t in thresholds:
        pck = torch.sum(j_err <= t) / j_err.shape[0]
        pck_curve.append(pck)
    auc = torch.trapz(torch.tensor(pck_curve), thresholds)
    auc /= norm_factor
    meta['joint_auc'] = auc.numpy()
    return meta

def mask_error(normalize_pred, gt):
    meta = {}    
    voxel_spacing = [0.5, 0.5, 0.5]
    connectivity = 1
    
    gt_expand = expand_as_one_hot(gt, 21)
    per_channel_dice = compute_per_channel_dice(normalize_pred, gt_expand)
    
    normalize_pred = torch.argmax(normalize_pred, dim=1).squeeze()
    # print(normalize_pred.shape)

    normalize_pred = normalize_pred.squeeze().cpu().detach().numpy()
    gt = gt.squeeze().cpu().detach().numpy()
    # batch_size = normalize_pred.shape[0]
    # for b in range(batch_size):

    hds = np.zeros(20)
    for i in range(1, 21):
#         print(i)
        if np.sum(gt==i) > 0:
            if np.sum(normalize_pred==i) == 0:
                hds[i-1] = 0
            else:
                hd = metric.binary.hd(normalize_pred==i, gt==i, voxel_spacing, connectivity)
                hds[i-1] = hd
    # hds = []
    # for i in range(1, 21):
    #     hd = metric.hd(normalize_pred==i, gt==i, voxel_spacing, connectivity)
    #     # print(hd)
    #     hds.append(hd)
    
    hds = np.stack(hds)
    print(hds)
    meta['per_channel_hd'] = hds
    meta['hd'] = np.mean(hds)

    meta['per_channel_dice'] = per_channel_dice.detach().cpu().numpy()
    meta['mean_dice'] = torch.mean(per_channel_dice[1:]).cpu().detach().numpy()
    return meta

class JoinLoss(_AbstractDiceLoss):

    def __init__(self, classes=21, skip_index_after=None, weight=None, sigmoid_normalization=True ):
        super().__init__(weight, sigmoid_normalization)

        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

        self.seg_loss = BCEDiceLoss(classes=classes)
        self.joint_loss = MSELoss()

    def forward(self, input, target, val=False):
        pred_mask, joint, posep, shapep = input
        target_mask, joint_gt, theta, beta = target
        
        meta = None

        loss_dice, per_channel_dice = self.seg_loss(pred_mask, target_mask)
        if joint is not None:
            loss_joint = self.joint_loss(joint, joint_gt) / 2500.
            if val:
                meta = self.get_joint_measures(joint, joint_gt)
                meta['loss_joint'] = loss_joint

            beta = beta[:, :shapep.shape[-1]]
            loss_beta = self.joint_loss(shapep, beta)
            loss_theta = self.joint_loss(posep, theta)
            loss = loss_dice + loss_joint + loss_beta + loss_theta
        else:
            loss = loss_dice



        return loss, per_channel_dice, meta



    def get_joint_measures(self, pred, gt):
        # joint evaluation
        meta = {}
        j_err = torch.sqrt(torch.sum((gt- pred)**2, axis=1))
        meta['joint_mse'] = torch.mean(j_err)

        thresholds = torch.linspace(0, 50, 10)
        norm_factor = torch.trapz(torch.ones_like(thresholds), thresholds)

        pck_curve = []
        j_err = j_err.view(-1)
        for t in thresholds:
            pck = torch.sum(j_err <= t) / j_err.shape[0]
            pck_curve.append(pck)
        auc = torch.trapz(torch.tensor(pck_curve), thresholds)
        auc /= norm_factor
        meta['joint_auc'] = auc
        return meta