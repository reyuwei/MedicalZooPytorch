import os
import numpy as np
import torch
from torch.nn import Module
from lib.bonepth import rodrigues_layer, rotproj
from lib.bonepth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                                     subtract_flat_id, make_list, th_scalemat_scale)
import lib.bonepth.globalvar as globalvar


class BoneLayer(Module):
    __constants__ = [
        'use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx'
    ]

    def __init__(self, template_ske: dict, center_idx=0, ncomps_shape=35, use_jreg=True, use_shapepca=True):
        super().__init__()
        self.center_idx = center_idx
        self.ncomps = 20 * 3
        self.bone_data = template_ske
        self.use_shapepca = use_shapepca
        self.ncomps_shape = ncomps_shape
        self.use_jreg = use_jreg

        self.register_buffer('th_v_normals',
                             torch.from_numpy(self.bone_data['v_template_normals']).float().unsqueeze(0))
        self.register_buffer('th_weights', torch.from_numpy(self.bone_data['weights']).float())
        self.register_buffer('th_faces', torch.from_numpy(self.bone_data['f'].astype(np.int32)).long())

        self.register_buffer('identity_rot', torch.eye(3).float())
        self.verts_separater = self.bone_data['verts_separater']
        self.faces_separater = self.bone_data['faces_separater']

        shape_pca_dict = np.load("lib/bonepth/shape_pca.pkl", allow_pickle=True)
        mean = shape_pca_dict['mean']
        basis = shape_pca_dict['basis']
        self.register_buffer('th_v_template', torch.from_numpy(mean.reshape(-1, 3)).float().unsqueeze(0))
        th_shapedirs = torch.from_numpy(basis[:self.ncomps_shape, ...].reshape(self.ncomps_shape, -1, 3)).float().permute(1, 2, 0)
            # torch.from_numpy(basis[:self.ncomps_shape, ...].reshape(self.ncomps_shape, -1, 3)).float().permute(1, 2, 0))
        self.register_buffer("th_shapedirs", th_shapedirs)
        self.register_buffer('th_betas', torch.zeros(self.ncomps_shape).float())

        if self.use_jreg:
            self.J_reg = np.load("lib/bonepth/J_regressor.pkl", allow_pickle=True)
            self.register_buffer("th_J_regressor", torch.from_numpy(self.J_reg).float())
            th_j = torch.matmul(self.th_J_regressor, self.th_v_template)
        else:
            th_j = torch.from_numpy(self.bone_data['joints']).float()

        self.register_buffer('th_joints', th_j.float().view(1, -1, 3))

        # Kinematic chain params
        kinetree = globalvar.JOINT_PARENT_ID_DICT
        self.kintree_parents = []
        for i in range(globalvar.STATIC_JOINT_NUM):
            self.kintree_parents.append(kinetree[i])

    def init_skeleton(self):
        if hasattr(self, "th_J_regressor"):
            return torch.matmul(self.th_J_regressor, self.th_v_template)
        else:
            return self.th_joints

    def forward(self, th_full_pose, th_offset=None, th_shape_param=None, th_scale=None, root_position=None):
        batch_size = th_full_pose.shape[0]

        if th_offset is not None:
            th_offset = th_offset.view(batch_size, -1, 3)

        if th_scale is not None:
            if th_scale.shape[1] > 1:
                th_global_scale = th_scale[:, :1]
                th_scale_bone_mat = th_scalemat_scale(th_scale[:, 1:])
            else:
                th_global_scale = th_scale
                th_scale_bone_mat = None
        else:
            th_global_scale = None
            th_scale_bone_mat = None

        if th_full_pose.dim() == 4:  # is mat
            assert th_full_pose.shape[2:4] == (3, 3), (
                'When not self.use_pca, th_pose_coeffs have 3x3 matrix for two'
                'last dims, got {}'.format(th_full_pose.shape[2:4]))
            th_full_pose = th_full_pose.view(batch_size, -1, 3, 3)
            th_rot_map = th_full_pose[:, 1:, :, :].view(batch_size, -1)
            root_rot = th_full_pose[:, 0, :, :].view(batch_size, 3, 3)
        elif th_full_pose.dim() == 3:  # axis-angle
            # compute rotation matrix from axis-angle while skipping global rotation
            th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose.view(batch_size, -1))
            th_full_pose = th_full_pose.view(batch_size, -1, 3)
            root_rot = rodrigues_layer.batch_rodrigues(th_full_pose[:, 0]).view(batch_size, 3, 3)


        if th_shape_param is None or self.use_shapepca is False:
            th_betas = self.th_betas.repeat(batch_size, 1)
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                    th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
        else:
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                    th_shape_param.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template


        if self.use_jreg:
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
        else:
            th_j = self.th_joints.repeat(batch_size, 1, 1)

        # Global rigid transformation
        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(globalvar.STATIC_JOINT_NUM - 1):
            i_val_joint = int(i + 1)
            joint_j = th_j[:, i_val_joint, :].contiguous().view(batch_size, 3, 1)
            joint_offset = None

            if i_val_joint in globalvar.JOINT_ID_BONE_DICT:
                i_val_bone = globalvar.JOINT_ID_BONE_DICT[i_val_joint]
                joint_rot = th_rot_map[:, (i_val_bone - 1) * 9:i_val_bone * 9].contiguous().view(batch_size, 3, 3)
                if th_offset is not None:
                    if i_val_bone in [1, 4, 8, 12, 16]:
                        joint_offset = th_offset[:, i_val_bone - 1, :].contiguous().view(batch_size, 3, 1)
            else:
                joint_rot = self.identity_rot.repeat(batch_size, 1, 1)

            parent = make_list(self.kintree_parents)[i_val_joint]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)

            if joint_offset is not None:
                joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_offset + joint_j - parent_j], 2))
            else:
                joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))

            if th_scale_bone_mat is not None:
                joint_rel_transform = torch.bmm(joint_rel_transform, th_scale_bone_mat[:, i_val_bone - 1])

            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))

        th_results_global = th_results
        th_results2 = torch.zeros((batch_size, 4, 4, globalvar.STATIC_JOINT_NUM),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(globalvar.STATIC_JOINT_NUM):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat([th_j[:, i], padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([th_v_shaped.transpose(2, 1),
                                     torch.ones((batch_size, 1, th_v_shaped.shape[1]), dtype=th_T.dtype,
                                                device=th_T.device), ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # scaling
        if th_global_scale is not None:
            center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint

            verts_scale = th_global_scale.expand(th_verts.shape[0], th_verts.shape[1])
            verts_scale = verts_scale.unsqueeze(2).repeat(1, 1, 3)
            th_verts = th_verts * verts_scale

            j_scale = th_global_scale.expand(th_jtr.shape[0], th_jtr.shape[1])
            j_scale = j_scale.unsqueeze(2).repeat(1, 1, 3)
            th_jtr = th_jtr * j_scale

        # translation
        if root_position is not None:
            root_position = root_position.view(batch_size, 1, 3)
            center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
            offset = root_position - center_joint
            th_jtr = th_jtr + offset
            th_verts = th_verts + offset

        return th_verts, th_jtr
