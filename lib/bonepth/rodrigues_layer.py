"""
This part reuses code from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py
which is part of a PyTorch port of SMPL.
Thanks to Zhang Xiong (MandyMo) for making this great code available on github !
"""

import torch


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                   2], norm_quat[:,
                                                       3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_get_axis_angle(vector):
    angle = torch.norm(vector, 2, 1)
    axes = vector / angle.unsqueeze(1)
    return axes, angle


def axisang2quat(axisang):
    # quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return quat


def quat2axisang(quat):
    # quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    # w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    w = norm_quat[:, 0]
    xyz = norm_quat[:, 1:]

    batch_size = quat.shape[0]
    angle = 2 * torch.acos(w)
    s = torch.sqrt(1 - w * w).view(batch_size, -1)
    xyzn = torch.zeros_like(xyz).to(xyz)
    for i in range(s.shape[0]):
        if s[i] >= 0.001:
            xyzn[i] = angle[i] * xyz[i] / s[i]
        else:
            xyzn[i] = xyz[i] * angle[i]
    # inmask = s >= 0.001
    # xyzn[inmask] = xyz / s
    # xyzn[~inmask] = xyz
    # axisang = xyzn * angle
    axisang = xyzn
    return axisang


def composite_axisang(a, b):
    # first a, then b
    # quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    quat_a = axisang2quat(a)
    quat_b = axisang2quat(b)
    # comp = quat_b * quat_a
    q = quat_b
    q_ = quat_a
    comp_xyz = torch.cross(q[:, 1:], q_[:, 1:])
    for i in range(q.shape[0]):
        comp_xyz[i] = torch.cross(q[i:i + 1, 1:], q_[i:i + 1, 1:]) + \
                        q[i:i + 1, 0] * q_[i:i + 1, 1:] + q_[i:i + 1, 0] * q[i:i + 1, 1:]

    comp_w = (q[:, 0] * q_[:, 0]).view(-1, 1) - (torch.bmm(q[:, 1:].unsqueeze(1), q_[:, 1:].unsqueeze(-1))).view(-1, 1)
    comp = torch.cat([comp_w, comp_xyz], dim=1)
    comp_axisang = quat2axisang(comp)
    return comp_axisang
