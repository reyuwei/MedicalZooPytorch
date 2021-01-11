from lib.losses3D.BCE_dice import BCEDiceLoss
from lib.losses3D.BaseClass import _AbstractDiceLoss
from lib.losses3D.basic import *
from torch.nn import MSELoss

class JoinLoss(_AbstractDiceLoss):

    def __init__(self, classes=21, skip_index_after=None, weight=None, sigmoid_normalization=True ):
        super().__init__(weight, sigmoid_normalization)

        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

        self.seg_loss = BCEDiceLoss(classes=classes)
        self.joint_loss = MSELoss()

    def forward(self, input, target):
        pred_mask, joint, _, _ = input
        target_mask, joint_gt = target

        loss_dice, per_channel_dice = self.seg_loss(pred_mask, target_mask)
        if joint is not None:
            loss_joint = self.joint_loss(joint, joint_gt)
            loss = loss_dice + loss_joint
        else:
            loss = loss_dice

        return loss, per_channel_dice