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

    def forward(self, input, target, val=False):
        pred_mask, joint, pose, shape = input
        target_mask, joint_gt = target
        meta = None
        loss_dice, per_channel_dice = self.seg_loss(pred_mask, target_mask)
        if joint is not None:
            loss_joint = self.joint_loss(joint, joint_gt) / 2500.
            if val:
                meta = self.get_joint_measures(joint, joint_gt)
                meta['loss_joint'] = loss_joint
            loss = loss_dice + loss_joint
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