import numpy as np
from .env_utils import (
    create_obs_config
)

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.environment import Environment
from rlbench.action_modes.gripper_action_modes import Discrete
import torch


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_mask=False,
        headless=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_mask = apply_mask
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = create_obs_config(
            image_size=image_size,
            apply_rgb=self.apply_rgb,
            apply_depth=self.apply_depth,
            apply_pc=self.apply_pc,
            apply_mask=self.apply_mask,
            apply_cameras=self.apply_cameras
        )
        
    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos
    
    def get_obs_action(self, obs):
        """
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": [], "mask":[]}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]
            
            if self.apply_mask:
                mask = getattr(obs, "{}_mask".format(cam))
                state_dict["mask"] += [mask]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

