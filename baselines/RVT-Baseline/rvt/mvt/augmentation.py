# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
import torch
import rvt.mvt.aug_utils as aug_utils
from scipy.spatial.transform import Rotation
import torch.nn.functional as F


def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation.
    :param pcd:
        Either:
        - list of point clouds [[bs, 3, H, W], ...] for N cameras
        - point cloud [bs, 3, H, W]
        - point cloud [bs, 3, num_point]
        - point cloud [bs, num_point, 3]
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds in the same format as input
    """
    # batch bounds if necessary

    # for easier compatibility
    single_pc = False
    if not isinstance(pcd, list):
        single_pc = True
        pcd = [pcd]

    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        permute_p = False
        if len(p.shape) == 3:
            if p_shape[-1] == 3:
                num_points = p_shape[-2]
                p = p.permute(0, 2, 1)
                permute_p = True
            elif p_shape[-2] == 3:
                num_points = p_shape[-1]
            else:
                assert False, p_shape

        elif len(p.shape) == 4:
            assert p_shape[-1] != 3, p_shape[-1]
            assert p_shape[-2] != 3, p_shape[-2]
            num_points = p_shape[-1] * p_shape[-2]

        else:
            assert False, len(p.shape)

        action_trans_3x1 = (
            action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )
        trans_shift_3x1 = (
            trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(
            p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
        ).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(
            action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
        )
        action_then_trans_3x1_y = torch.clamp(
            action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
        )
        action_then_trans_3x1_z = torch.clamp(
            action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
        )
        action_then_trans_3x1 = torch.stack(
            [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
            dim=1,
        )

        # shift back the origin
        perturbed_p_flat_3x1 = (
            perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
        )
        if permute_p:
            perturbed_p_flat_3x1 = torch.permute(perturbed_p_flat_3x1, (0, 2, 1))
        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)

    if single_pc:
        perturbed_pcd = perturbed_pcd[0]

    return perturbed_pcd

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def apply_se3_augmentation(
    pcd,
    action_gripper_pose,
    action_trans,
    action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
):
    """Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat(
        (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
    )
    action_gripper_rot = quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.0)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * aug_utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = aug_utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = aug_utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = aug_utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        )
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = matrix_to_quaternion(
            perturbed_action_gripper_4x4[:, :3, :3]
        )
        perturbed_action_quat_xyzw = torch.cat(
            [
                perturbed_action_quat_wxyz[:, 1:],
                perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
            ],
            dim=1,
        ).cpu().numpy()

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = aug_utils.point_to_voxel_index(
                perturbed_action_trans[b], voxel_size, bounds_np
            )
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = aug_utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = aug_utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(
                disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(
            device=device
        )

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return action_trans, action_rot_grip, pcd


def apply_se3_aug_con(
    pcd,
    action_gripper_pose,
    bounds,
    trans_aug_range,
    rot_aug_range,
    scale_aug_range=False,
    single_scale=True,
    ver=2,
):
    """Apply SE3 augmentation to a point clouds and actions.
    :param pcd: [bs, num_points, 3]
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param bounds: metric scene bounds
        Either:
        - [bs, 6]
        - [6]
    :param trans_aug_range: range of translation augmentation
        [x_range, y_range, z_range]; this is expressed as the percentage of the scene bound
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param scale_aug_range: range of scale augmentation [x_range, y_range, z_range]
    :param single_scale: whether we preserve the relative dimensions
    :return: perturbed action_gripper_pose,  pcd
    """

    # batch size
    bs = pcd.shape[0]
    device = pcd.device

    if len(bounds.shape) == 1:
        bounds = bounds.unsqueeze(0).repeat(bs, 1).to(device)
    if len(trans_aug_range.shape) == 1:
        trans_aug_range = trans_aug_range.unsqueeze(0).repeat(bs, 1).to(device)
    if len(rot_aug_range.shape) == 1:
        rot_aug_range = rot_aug_range.unsqueeze(0).repeat(bs, 1)

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]

    if ver == 1:
        action_gripper_quat_wxyz = torch.cat(
            (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
        )
        action_gripper_rot = quaternion_to_matrix(action_gripper_quat_wxyz)

    elif ver == 2:
        # applying gimble fix to calculate a new action_gripper_rot
        r = Rotation.from_quat(action_gripper_pose[:, 3:7].cpu().numpy())
        euler = r.as_euler("xyz", degrees=True)
        euler = aug_utils.sensitive_gimble_fix(euler)
        action_gripper_rot = torch.tensor(
            Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
            device=action_gripper_pose.device,
        )
    else:
        assert False

    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    # sample translation perturbation with specified range
    # augmentation range is a percentage of the scene bound
    trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
    # rand_dist samples value from -1 to 1
    trans_shift = trans_range * aug_utils.rand_dist((bs, 3)).to(device=device)

    # apply bounded translations
    bounds_x_min, bounds_x_max = bounds[:, 0], bounds[:, 3]
    bounds_y_min, bounds_y_max = bounds[:, 1], bounds[:, 4]
    bounds_z_min, bounds_z_max = bounds[:, 2], bounds[:, 5]

    trans_shift[:, 0] = torch.clamp(
        trans_shift[:, 0],
        min=bounds_x_min - action_gripper_trans[:, 0],
        max=bounds_x_max - action_gripper_trans[:, 0],
    )
    trans_shift[:, 1] = torch.clamp(
        trans_shift[:, 1],
        min=bounds_y_min - action_gripper_trans[:, 1],
        max=bounds_y_max - action_gripper_trans[:, 1],
    )
    trans_shift[:, 2] = torch.clamp(
        trans_shift[:, 2],
        min=bounds_z_min - action_gripper_trans[:, 2],
        max=bounds_z_max - action_gripper_trans[:, 2],
    )

    trans_shift_4x4 = identity_4x4.detach().clone()
    trans_shift_4x4[:, 0:3, 3] = trans_shift

    roll = np.deg2rad(rot_aug_range[:, 0:1] * aug_utils.rand_dist((bs, 1)))
    pitch = np.deg2rad(rot_aug_range[:, 1:2] * aug_utils.rand_dist((bs, 1)))
    yaw = np.deg2rad(rot_aug_range[:, 2:3] * aug_utils.rand_dist((bs, 1)))
    rot_shift_3x3 = euler_angles_to_matrix(
        torch.cat((roll, pitch, yaw), dim=1), "XYZ"
    )
    rot_shift_4x4 = identity_4x4.detach().clone()
    rot_shift_4x4[:, :3, :3] = rot_shift_3x3

    if ver == 1:
        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
    elif ver == 2:
        perturbed_action_gripper_4x4 = identity_4x4.detach().clone()
        perturbed_action_gripper_4x4[:, 0:3, 3] = action_gripper_4x4[:, 0:3, 3]
        perturbed_action_gripper_4x4[:, :3, :3] = torch.bmm(
            rot_shift_4x4.transpose(1, 2)[:, :3, :3], action_gripper_4x4[:, :3, :3]
        )
    else:
        assert False

    perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

    # convert transformation matrix to translation + quaternion
    perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
    perturbed_action_quat_wxyz = matrix_to_quaternion(
        perturbed_action_gripper_4x4[:, :3, :3]
    )
    perturbed_action_quat_xyzw = (
        torch.cat(
            [
                perturbed_action_quat_wxyz[:, 1:],
                perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
            ],
            dim=1,
        )
        .cpu()
        .numpy()
    )

    # TODO: add scale augmentation

    # apply perturbation to pointclouds
    # takes care for not moving the point out of the image
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return perturbed_action_trans, perturbed_action_quat_xyzw, pcd



def apply_se3_aug_con_with_sub_goal(
    pcd,
    action_gripper_pose,
    sub_goal_action_gripper_pose,
    bounds,
    trans_aug_range,
    rot_aug_range,
    scale_aug_range=False,
    single_scale=True,
    ver=2,
):
    """
    Apply SE3 augmentation to point clouds, main action pose, and sub-goal pose.
    """
    bs = pcd.shape[0]
    device = pcd.device

    if len(bounds.shape) == 1:
        bounds = bounds.unsqueeze(0).repeat(bs, 1).to(device)
    if len(trans_aug_range.shape) == 1:
        trans_aug_range = trans_aug_range.unsqueeze(0).repeat(bs, 1).to(device)
    if len(rot_aug_range.shape) == 1:
        rot_aug_range = rot_aug_range.unsqueeze(0).repeat(bs, 1)

    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device)

    def pose_to_matrix(pose, ver):
        trans = pose[:, :3]
        if ver == 1:
            quat_wxyz = torch.cat((pose[:, 6:7], pose[:, 3:6]), dim=1)
            rot = quaternion_to_matrix(quat_wxyz)
        elif ver == 2:
            r = Rotation.from_quat(pose[:, 3:7].cpu().numpy())
            euler = aug_utils.sensitive_gimble_fix(r.as_euler("xyz", degrees=True))
            rot = torch.tensor(
                Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
                device=pose.device,
            )
        else:
            raise ValueError("Unsupported ver")
        mat = identity_4x4.clone()
        mat[:, :3, :3] = rot
        mat[:, :3, 3] = trans
        return mat

    action_mat = pose_to_matrix(action_gripper_pose, ver)
    sub_goal_mat = pose_to_matrix(sub_goal_action_gripper_pose, ver)

    # translation perturbation
    trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
    trans_shift = trans_range * aug_utils.rand_dist((bs, 3)).to(device=device)

    for i in range(3):
        trans_shift[:, i] = torch.clamp(
            trans_shift[:, i],
            min=bounds[:, i] - action_gripper_pose[:, i],
            max=bounds[:, i + 3] - action_gripper_pose[:, i],
        )

    trans_shift_4x4 = identity_4x4.clone()
    trans_shift_4x4[:, :3, 3] = trans_shift

    # rotation perturbation
    roll = np.deg2rad(rot_aug_range[:, 0:1] * aug_utils.rand_dist((bs, 1)))
    pitch = np.deg2rad(rot_aug_range[:, 1:2] * aug_utils.rand_dist((bs, 1)))
    yaw = np.deg2rad(rot_aug_range[:, 2:3] * aug_utils.rand_dist((bs, 1)))
    rot_shift_3x3 = euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
    rot_shift_4x4 = identity_4x4.clone()
    rot_shift_4x4[:, :3, :3] = rot_shift_3x3

    # apply SE3 perturbation to action and subgoal
    if ver == 1:
        new_action_mat = torch.bmm(action_mat, rot_shift_4x4)
        new_subgoal_mat = torch.bmm(sub_goal_mat, rot_shift_4x4)
    elif ver == 2:
        new_action_mat = identity_4x4.clone()
        new_action_mat[:, :3, 3] = action_mat[:, :3, 3]
        new_action_mat[:, :3, :3] = torch.bmm(
            rot_shift_4x4[:, :3, :3].transpose(1, 2), action_mat[:, :3, :3]
        )

        new_subgoal_mat = identity_4x4.clone()
        new_subgoal_mat[:, :3, 3] = sub_goal_mat[:, :3, 3]
        new_subgoal_mat[:, :3, :3] = torch.bmm(
            rot_shift_4x4[:, :3, :3].transpose(1, 2), sub_goal_mat[:, :3, :3]
        )
    else:
        raise ValueError("Unsupported ver")

    new_action_mat[:, :3, 3] += trans_shift
    new_subgoal_mat[:, :3, 3] += trans_shift

    def mat_to_pose(mat):
        trans = mat[:, :3, 3].cpu().numpy()
        quat_wxyz = matrix_to_quaternion(mat[:, :3, :3])
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]], dim=1).cpu().numpy()
        return trans, quat_xyzw

    new_action_trans, new_action_quat = mat_to_pose(new_action_mat)
    new_subgoal_trans, new_subgoal_quat = mat_to_pose(new_subgoal_mat)

    # transform point cloud
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_mat, bounds)

    return new_action_trans, new_action_quat, new_subgoal_trans, new_subgoal_quat, pcd