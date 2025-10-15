from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode
import numpy as np
from scipy.spatial.transform import Rotation as R
from rlbench.backend.observation import Observation
from typing import List, Tuple
import torch
import torch.nn.functional as F

def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class

def create_obs_config(
        image_size, apply_rgb, apply_depth, apply_pc, apply_mask, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=apply_mask,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        llm_view = apply_cameras
        kwargs = {}
        for n in llm_view:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=True,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=True,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config

def convert_proprio_to_euler(proprio):
    ee_position = proprio[:3]
    ee_orientation = proprio[3:-1]

    ee_rotation = R.from_quat(ee_orientation)
    ee_euler_orientation = ee_rotation.as_euler("xyz")

    return np.concatenate(
        [ee_position, ee_euler_orientation, [proprio[-1]]],
        axis=-1,
        dtype=np.float64,
    )

def _is_stopped(low_dim_obs: List[Observation], i, stopped_buffer, delta):
    """判断机器人是否停止运动
    
    Args:
        low_dim_obs: RLBench观测序列
        i: 当前时间步
        stopped_buffer: 停止缓冲计数器
        delta: 速度阈值
    """
    next_is_not_final = i == (len(low_dim_obs) - 2)
    
    gripper_state_no_change = i < (len(low_dim_obs) - 2) and (
        low_dim_obs[i].gripper_open == low_dim_obs[i + 1].gripper_open
        and low_dim_obs[i].gripper_open == low_dim_obs[max(0, i - 1)].gripper_open
        and low_dim_obs[max(0, i - 2)].gripper_open == low_dim_obs[max(0, i - 1)].gripper_open
    )
    
    small_delta = np.allclose(low_dim_obs[i].joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped

def keypoint_discovery(low_dim_obs: List[Observation], stopping_delta=0.1) -> List[int]:
    """发现轨迹中的关键点
    
    Args:
        low_dim_obs: RLBench观测序列
        stopping_delta: 判断停止的速度阈值
        
    Returns:
        episode_keypoints: 关键点索引列表
    """
    episode_keypoints = []
    prev_gripper_open = low_dim_obs[0].gripper_open
    stopped_buffer = 0

    for i in range(len(low_dim_obs)):
        stopped = _is_stopped(low_dim_obs, i, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # 如果夹持器状态改变或到达序列末尾
        last = i == (len(low_dim_obs) - 1)
        if i != 0 and (low_dim_obs[i].gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = low_dim_obs[i].gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints

def get_key_steps(low_dim_obs: List[Observation], velocity_threshold: float = 0.1) -> Tuple[List[int], List[Observation]]:
    """选择观测序列中的关键帧
    
    Args:
        low_dim_obs: RLBench观测序列
        velocity_threshold: 判断机器人停止的关节速度阈值
        
    Returns:
        key_steps: 选中的关键帧索引列表
        now_low_dim_obs: 对应的观测列表
    """
    # 获取关键帧
    keyframes = keypoint_discovery(low_dim_obs)
    if len(keyframes) == 0:
        keyframes = [0, len(low_dim_obs)-1]
    
    # 确保包含第一帧
    if keyframes[0] != 0:
        keyframes.insert(0, 0)
    
    # 确保包含最后一帧
    if keyframes[-1] != len(low_dim_obs)-1:
        keyframes.append(len(low_dim_obs)-1)
    
    # 提取对应的观测
    now_low_dim_obs = [low_dim_obs[i] for i in keyframes]
    
    return keyframes, now_low_dim_obs

def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)

def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v