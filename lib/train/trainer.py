import numpy as np
import torch
import medpy
from lib.losses3D.JoinLoss import joint_error, mask_error
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent
from lib.utils.evalutil import EvalUtil

class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, fortest=False):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)


        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1

        if self.args.resume != "":
            self._resume_checkpoint(self.args.resume)

    def _resume_checkpoint(self,resume_folder):
        from pathlib import Path
        # ckpt_file = Path(resume_folder) / (Path(resume_folder).stem + "_last_epoch.pth")
        ckpt_file = Path(resume_folder) / (Path(resume_folder).stem + "_last_epoch.pth")
        self.epoch, self.optimizer = self.model.restore_checkpoint(ckpt_file, optimizer=self.optimizer)

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):
            self.maybe_update_lr(epoch)
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer)

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            # input_tensor.requires_grad = True
            output = self.model(input_tensor)
            loss_dice, per_ch_score, _ = self.criterion(output, target, val=False)
            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                # input_tensor.requires_grad = False

                output = self.model(input_tensor)
                loss, per_ch_score, meta = self.criterion(output, target, val=True)
                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx, meta)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

    def test(self):
        self.model.eval()
        seg_output = torch.FloatTensor()
        seg_gt = torch.FloatTensor()
        joint_output = torch.FloatTensor()
        joint_gt = torch.FloatTensor()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                print(batch_idx, len(self.valid_data_loader))
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                output = self.model(input_tensor)
                
                pred_mask, pred_joint, pose_param, shape_param = output
                target_mask, target_joint = target

                seg_output = torch.cat([seg_output, pred_mask.detach().cpu()], dim=0)
                seg_gt = torch.cat([seg_gt, target_mask.detach().cpu()], dim=0)
                if pred_joint is not None:
                    joint_output = torch.cat([joint_output, pred_joint.detach().cpu()], dim=0)
                    joint_gt = torch.cat([joint_gt, target_joint.detach().cpu()], dim=0)
        
        normalization = torch.nn.Sigmoid()
        normalize_seg_output = normalization(seg_output)

        meta = {}
        mask_meta = mask_error(normalize_seg_output, seg_gt)
        meta.update(mask_meta)
        if joint_gt.shape[0] > 0:
            joint_meta = joint_error(joint_output, joint_gt)
            meta.update(joint_meta)

        return meta, (joint_output, joint_gt), (normalize_seg_output, seg_gt)
        


    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.args.nEpochs, self.args.lr, 0.9)
        self.writer.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)