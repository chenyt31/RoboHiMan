# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult
from pyrep.objects.dummy import Dummy
from src.high_level.vlm.agent import MultiModalAgent
from pathlib import Path
from typing import List, Dict
import torchvision
from clip import tokenize

class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0', vlm_agent:MultiModalAgent=None, tmp_dir:Path=None):
        self._env_device = env_device
        self.vlm_agent :MultiModalAgent = vlm_agent
        self.tmp_dir = tmp_dir

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
    
    def _save_images(self, obs:Dict[str, torch.Tensor], base_target_cameras=['front', 'wrist'], step:int=0):
        # save image for vlm input
        base_images = []
        for target_cam in base_target_cameras:
            img_tensor = obs["%s_rgb" % target_cam][0][0].float() / 255.0 # numpy array, uint8, shape (H, W, 3)
            base_images.append(img_tensor)
        base_path: Path = Path(self.tmp_dir) / "images" / f"{step}"
        base_frame_images = []
        base_path.mkdir(parents=True, exist_ok=True)
        for j, img in enumerate(base_images):
            torchvision.utils.save_image(img, base_path / f"{j}.png")
            base_frame_images.append(base_path / f"{j}.png")
        return base_frame_images

    def _get_home_pose(self):
        home_pose = np.array(Dummy("Panda_tip").get_pose())
        return np.concatenate((home_pose, [1, 0]))

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  eval_mode:str = "vanilla"):
        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        # ============================================================================
        home_pose = self._get_home_pose()
        instr_index = 0
        grasped_objects = env._rlbench_env._scene.robot.gripper.get_grasped_objects()  # noqa: SLF001
        prev_grasped_objects_len = len(grasped_objects)
        prev_gripper_action = 1.0 
        half_descriptions = env.descriptions["oracle_half"][0].split('\n')
        return_home = False
        need_new_subtask = True 
        history_instr = []
        # ============================================================================
        for step in range(episode_length):
            print(f'step:{step}, return_home:{return_home}, instr:{half_descriptions[instr_index]}')
            if return_home:
                act_result = ActResult(home_pose)
                return_home = False
            else:
                prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}

                if eval_mode == "half":
                    prepped_data["lang_goal_tokens"] = prepped_data["half_lang_goal_tokens"][:, :, instr_index, :]
                elif eval_mode == "vlm":
                    if need_new_subtask:
                        base_frame_images = self._save_images(prepped_data, step=step)
                        task_description = env.descriptions["vanilla"][0]
                        variables={
                            "image_paths": base_frame_images,
                            "task_description": task_description,
                        }
                        
                        # TODO: history may contain current sub-task because it may not be performed yet
                        # variables["pre_sub_tasks"] = history_instr

                        result = self.vlm_agent.run(
                            task_name="next_sub_task",
                            variables=variables,
                            fields=("reasoning", "sub_task")
                        )[0]
                        try:
                            reasoning = result['reasoning'][0]
                        except:
                            reasoning = ""
                        try:
                            sub_task = result['sub_task'][0]
                            history_instr.append(sub_task)
                            need_new_subtask = False
                        except:
                            sub_task = ""
                        print(f'step:{step}, vlm: {sub_task}')
                        prepped_data["lang_goal_tokens"] = torch.tensor(np.array([tokenize([sub_task])[0].numpy()]), device=self._env_device).unsqueeze(0)
                    else:
                        # 沿用上一次生成的 sub_task
                        sub_task = history_instr[-1] if history_instr else ""
                        prepped_data["lang_goal_tokens"] = torch.tensor(
                            np.array([tokenize([sub_task])[0].numpy()]),
                            device=self._env_device
                        ).unsqueeze(0)
                else:
                    pass
                    
                if not replay_ground_truth:
                    act_result = agent.act(step_signal.value, prepped_data,
                                        deterministic=eval)
                else:
                    if step >= len(actions):
                        return
                    act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
            
            # plan forward according to current task
            # ============================================================================
            grasped_objects = env._rlbench_env._scene.robot.gripper.get_grasped_objects()  # noqa: SLF001
            cur_grasped_objects_len = len(grasped_objects)
            cur_gripper_action = act_result.action[-2]
            print(f"step:{step}, prev_gripper_action:{prev_gripper_action}, cur_gripper_action:{cur_gripper_action}")

            if (cur_grasped_objects_len != prev_grasped_objects_len or 
                abs(cur_gripper_action - prev_gripper_action) > 1e-6):
                instr_index = (instr_index + 1) % len(half_descriptions)
                need_new_subtask = True

                # reset when grasped changed and gripper open
                if abs(act_result.action[-2] - 1.000) < 1e-9:
                    return_home = True
                    env._i = -1
            prev_grasped_objects_len = cur_grasped_objects_len
            prev_gripper_action = cur_gripper_action
            # ============================================================================