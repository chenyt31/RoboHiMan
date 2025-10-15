import blosc
import pickle

import einops
from pickle import UnpicklingError
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from typing import Dict, List, Optional, Sequence
from pathlib import Path
Instructions_feature = Dict[str, Dict[int, torch.Tensor]]
Instructions = Dict[str, Dict[int, List[str]]]


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None

def load_instructions_feature(
    instructions_feature: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions_feature]:
    if instructions_feature is not None:
        with open(instructions_feature, "rb") as fid:
            data: Instructions_feature = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None

def load_instructions(
    instructions: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None

def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat

def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans

def trans_pc(pc, loc, sca):
    """
    change location of the center of the pc and scale it
    :param pc:
        either:
        - tensor of shape(b, num_points, 3)
        - tensor of shape(b, 3)
        - list of pc each with size (num_points, 3)
    :param loc: (b, 3 )
    :param sca: 1 or (3)
    """
    assert len(loc.shape) == 2
    assert loc.shape[-1] == 3
    if isinstance(pc, list):
        assert all([(len(x.shape) == 2) and (x.shape[1] == 3) for x in pc])
        pc = [sca * (x - y) for x, y in zip(pc, loc)]
    elif isinstance(pc, torch.Tensor):
        assert len(pc.shape) in [2, 3]
        assert pc.shape[-1] == 3
        if len(pc.shape) == 2:
            pc = sca * (pc - loc)
        else:
            pc = sca * (pc - loc.unsqueeze(1))
    else:
        assert False

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        assert isinstance(x, torch.Tensor)
        return (x / sca) + loc

    return pc, rev_trans

def add_uni_noi(x, u):
    """
    adds uniform noise to a tensor x. output is tensor where each element is
    in [x-u, x+u]
    :param x: tensor
    :param u: float
    """
    assert isinstance(u, float)
    # move noise in -1 to 1
    noise = (2 * torch.rand(*x.shape, device=x.device)) - 1
    x = x + (u * noise)
    return x

class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_w - resized_size[1], 0), max(
                raw_h - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == (trajectory.shape[1] - 1):  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled
