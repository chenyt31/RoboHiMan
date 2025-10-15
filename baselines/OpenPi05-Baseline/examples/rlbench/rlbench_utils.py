import collections

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from pyrep.const import RenderMode
from rlbench.backend.observation import Observation
from rlbench.observation_config import CameraConfig
from rlbench.observation_config import ObservationConfig
from scipy.spatial.transform import Rotation


def create_obs_config(image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs):
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
        mask=False,
        image_size=image_size,
        render_mode=RenderMode.OPENGL,
        **kwargs,
    )

    camera_names = apply_cameras
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
    return ObservationConfig(
        front_camera=kwargs.get("front", unused_cams),
        left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
        right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
        wrist_camera=kwargs.get("wrist", unused_cams),
        overhead_camera=kwargs.get("overhead", unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )


def convert_proprio_to_euler(proprio):
    ee_position = proprio[:3]
    ee_orientation = proprio[3:-1]

    ee_rotation = Rotation.from_quat(ee_orientation)
    ee_euler_orientation = ee_rotation.as_euler("xyz")

    return np.concatenate(
        [ee_position, ee_euler_orientation, [proprio[-1]]],
        axis=-1,
        dtype=np.float64,
    )


class Actioner:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, resize_size: int = 224, replan_steps: int = 5):
        """
        Initialize Actioner
        Args:
            cfg: Configuration object
            model: Pre-trained model
            processor: Processor (for OpenVLA)
            action_dim: Action dimension
        """
        self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        self.action_plan = collections.deque()
        self.replan_steps = replan_steps
        self.resize_size = resize_size

    def reset(self):
        self.action_plan.clear()

    def predict(self, proprio, img, wrist_img, instruction, **kwargs):
        """
        Predict next action
        Args:
            proprio: proprioceptive states
            images: RGB image
            text_embeds: instruction embeddings
            kwargs: Additional parameters about the task
        Returns:
            actions: Predicted actions
        """
        # Get preprocessed image
        img = np.ascontiguousarray(img)
        wrist_img = np.ascontiguousarray(wrist_img)
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, self.resize_size, self.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size)
        )
        if not self.action_plan:
            # Prepare observations dict
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": proprio,
                "prompt": instruction,
            }

            # Query model to get action
            action_chunk = self.client.infer(element)["actions"]
            assert len(action_chunk) >= self.replan_steps, (
                f"We want to replan every {self.replan_steps} steps, "
                f"but policy only predicts {len(action_chunk)} steps."
            )
            if self.replan_steps == -1:
                return action_chunk
            self.action_plan.extend(action_chunk[: self.replan_steps])

        # Get the next action
        action = self.action_plan.popleft()
        return [action]


def _is_stopped(low_dim_obs: list[Observation], i, stopped_buffer, delta):
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
    return stopped_buffer <= 0 and small_delta and (not next_is_not_final) and gripper_state_no_change


def keypoint_discovery(low_dim_obs: list[Observation], stopping_delta=0.1) -> list[int]:
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

    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == episode_keypoints[-2]:
        episode_keypoints.pop(-2)

    return episode_keypoints


def get_key_steps(
    low_dim_obs: list[Observation], velocity_threshold: float = 0.1
) -> tuple[list[int], list[Observation]]:
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
        keyframes = [0, len(low_dim_obs) - 1]

    # 确保包含第一帧
    if keyframes[0] != 0:
        keyframes.insert(0, 0)

    # 确保包含最后一帧
    if keyframes[-1] != len(low_dim_obs) - 1:
        keyframes.append(len(low_dim_obs) - 1)

    # 提取对应的观测
    now_low_dim_obs = [low_dim_obs[i] for i in keyframes]

    return keyframes, now_low_dim_obs
