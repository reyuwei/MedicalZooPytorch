import torch

from lib.bonepth import rodrigues_layer


def th_posemap_axisang(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb - 1):
        joint_idx_val = joint_idx + 1
        axis_ang = pose_vectors[:, joint_idx_val * 3:(joint_idx_val + 1) * 3]
        rot_mat = rodrigues_layer.batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    # rot_mats = torch.stack(rot_mats, 1).view(-1, 15 *9)
    rot_mats = torch.cat(rot_mats, 1)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats


def th_scalemat_scale(th_scale_bone):
    batch_size = th_scale_bone.shape[0]
    th_scale_bone_mat = torch.eye(4).repeat([batch_size, th_scale_bone.shape[-1], 1, 1])
    th_scale_bone_mat = th_scale_bone_mat.type_as(th_scale_bone).to(th_scale_bone.device)
    for s in range(th_scale_bone.shape[-1]):
        th_scale_bone_mat[:, s, 0, 0] = th_scale_bone[:, s]
        th_scale_bone_mat[:, s, 1, 1] = th_scale_bone[:, s]
        th_scale_bone_mat[:, s, 2, 2] = th_scale_bone[:, s]
    return th_scale_bone_mat


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
        rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results


def make_list(tensor):
    # type: (List[int]) -> List[int]
    return tensor
