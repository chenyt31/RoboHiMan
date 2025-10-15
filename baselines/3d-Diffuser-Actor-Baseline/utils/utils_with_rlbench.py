import os
import glob
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf
import open3d
import traceback
import torchvision
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops
import transformers

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.backend.task import Task
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper, ActionMode
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from pyrep.objects import VisionSensor, Dummy

from typing import Optional, Type, List, Dict, Any
import os
import json
from omegaconf import DictConfig, OmegaConf
from colosseum import (
    ASSETS_CONFIGS_FOLDER,
    ASSETS_JSON_FOLDER,
    TASKS_TTM_FOLDER,
    ASSETS_ATOMIC_CONFIGS_FOLDER,
    ASSETS_ATOMIC_JSON_FOLDER,
    ATOMIC_TASKS_TTM_FOLDER,
    ASSETS_COMPOSITIONAL_CONFIGS_FOLDER,
    ASSETS_COMPOSITIONAL_JSON_FOLDER,
    COMPOSITIONAL_TASKS_TTM_FOLDER
)
from colosseum import TASKS_PY_FOLDER, ATOMIC_TASKS_PY_FOLDER, COMPOSITIONAL_TASKS_PY_FOLDER

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import (
    ObservationConfigExt,
    check_and_make,
    name_to_class,
    save_demo,
)
from colosseum.variations.utils import safeGetValue
from functools import reduce
from src.high_level.vlm.agent import MultiModalAgent

def change_case(str):
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, str).lower()


def get_spreadsheet_config(
    base_cfg: DictConfig, collection_cfg: Dict[str, Any], spreadsheet_idx: int
) -> DictConfig:
    """
    Creates a new config object based on a base configuration, updated with
    entries to match the options from the data collection strategy in JSON
    format for the given spreadsheet index.

    Parameters
    ----------
        base_cfg : DictConfig
            The base configuration for the current task
        collection_cfg : Dict[str, Any]
            The data collection strategy parsed from the JSON strategy file
        spreadsheet_idx : int
            The index in the spreadsheet to use for the current task variation

    Returns
    -------
        DictConfig
            The new configuration object with the updated options for this
            variation
    """
    spreadsheet_cfg = base_cfg.copy()

    collections_variation_cfg = collection_cfg["strategy"][spreadsheet_idx][
        "variations"
    ]
    for collection_var_cfg in collections_variation_cfg:
        var_type = collection_var_cfg["type"]
        var_name = collection_var_cfg["name"]
        var_enabled = collection_var_cfg["enabled"]
        for variation_cfg in spreadsheet_cfg.env.scene.factors:
            if variation_cfg.variation != var_type:
                continue
            else:
                if var_name == "any" or (
                    "name" in variation_cfg and variation_cfg.name == var_name
                ):
                    variation_cfg.enabled = var_enabled

    return spreadsheet_cfg

class MultiTaskRLBenchEnv():

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 headless=True,
                 swap_task_every: int = 1,
                 base_cfg_name=None,
                 task_class_variation_idx=None):
        super(MultiTaskRLBenchEnv, self).__init__()

        self._task_classes = task_classes
        self._observation_config = observation_config
        # self._rlbench_env = Environment(
        #     action_mode=action_mode, obs_config=observation_config,
        #     dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ''
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every
        
        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self._base_cfg_name = base_cfg_name
        self._task_class_variation_idx = task_class_variation_idx
        self._action_mode = action_mode
        self._observation_config = observation_config
        self._dataset_root = dataset_root
        self._headless = headless

        self._record_cam = None
        self._recorded_images = []

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self.descriptions = descriptions
        try:
            self._lang_goal = descriptions['vanilla'][0]
        except:
            self._lang_goal = descriptions[0]
        # self._lang_goal = descriptions[0] # first description variant

    def launch(self, task_type=None):
        if task_type == "atomic":
            ASSETS_CONFIGS_FOLDER = ASSETS_ATOMIC_CONFIGS_FOLDER
            ASSETS_JSON_FOLDER = ASSETS_ATOMIC_JSON_FOLDER
            TASKS_TTM_FOLDER = ATOMIC_TASKS_TTM_FOLDER
        elif task_type == "compositional":
            ASSETS_CONFIGS_FOLDER = ASSETS_COMPOSITIONAL_CONFIGS_FOLDER
            ASSETS_JSON_FOLDER = ASSETS_COMPOSITIONAL_JSON_FOLDER
            TASKS_TTM_FOLDER = COMPOSITIONAL_TASKS_TTM_FOLDER
            
        base_cfg_path = os.path.join(ASSETS_CONFIGS_FOLDER, f"{self._base_cfg_name[self._active_task_id]}.yaml")
        if os.path.exists(base_cfg_path):
            with open(base_cfg_path, 'r') as f:
                base_cfg = OmegaConf.load(f)

        collection_cfg_path: str = (
        os.path.join(ASSETS_JSON_FOLDER, base_cfg.env.task_name) + ".json"
        )
        collection_cfg: Optional[Any] = None
        with open(collection_cfg_path, "r") as fh:
            collection_cfg = json.load(fh)

        if collection_cfg is None:
            return 1

        if "strategy" not in collection_cfg:
            return 1

        num_spreadsheet_idx = len(collection_cfg["strategy"])
        
        if self._task_class_variation_idx != None:
            full_config = get_spreadsheet_config(
                        base_cfg,
                        collection_cfg,
                        self._task_class_variation_idx[self._active_task_id],
                    )
            _, env_cfg = full_config.data, full_config.env  
        else:
            env_cfg = None

        self._rlbench_env = EnvironmentExt(
            action_mode=self._action_mode, obs_config=self._observation_config, 
            path_task_ttms=TASKS_TTM_FOLDER,
            dataset_root=self._dataset_root, headless=self._headless, env_config=env_cfg,)
        self._rlbench_env

        self._rlbench_env.launch()
        self._set_new_task()

        # record
        self._task._scene.register_step_callback(self._my_callback)
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_base = Dummy('cam_cinematic_base')
        cam_base.rotate([0, 0, np.pi * 0.75])
        self._record_cam = VisionSensor.create([320, 180])
        self._record_cam.set_explicit_handling(True)
        self._record_cam.set_pose(cam_placeholder.get_pose())
        self._record_cam.set_render_mode(RenderMode.OPENGL)

    def _my_callback(self):
        self._record_cam.handle_explicitly()
        cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(cap)

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self.descriptions = descriptions
        try:
            self._lang_goal = descriptions['vanilla'][0]
        except:
            self._lang_goal = descriptions[0]
        # self._lang_goal = descriptions[0] # first description variant

        return descriptions, obs
    
    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

        vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
        return vid

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0

        if self._task_class_variation_idx != None:
            self._task.set_variation(-1)
            self._task._task.task_path = self._task._task.name + f"_{str(self._task_class_variation_idx[self._active_task_id])}"
        else:
            self._task.set_variation(-1)

        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        self._recorded_images.clear()

        descriptions, obs = self._task.reset_to_demo(d)
        self.descriptions = descriptions
        try:
            self._lang_goal = descriptions['vanilla'][0]
        except:
            self._lang_goal = descriptions[0]
        return descriptions, obs



def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)


class Mover:

    def __init__(self, task, disabled=False, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action, collision_checking=False):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True,
        model_max_length=53
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._instr = None
        self._task_str = None

        self._policy.eval()
        self.model_max_length = model_max_length

        self.model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(self.device)
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer.model_max_length = model_max_length
        

    def load_episode(self, task_str, _lang_goal):
        self._task_str = task_str
        tokens = self.tokenizer([_lang_goal], padding="max_length")["input_ids"]
        lengths = [len(t) for t in tokens]
        if any(l > self.model_max_length for l in lengths):
            raise RuntimeError(f"Too long instructions: {lengths}")
        tokens = torch.tensor(tokens).to(self.device)
        with torch.no_grad():
            pred = self.model(tokens).last_hidden_state
        self._instr = pred.cpu()

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, rgbs, pcds, gripper,
                interpolation_length=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None, "trajectory": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(rgbs.device)

        # Predict trajectory
        if self._predict_trajectory:
            fake_traj = torch.full(
                [1, interpolation_length - 1, gripper.shape[-1]], 0
            ).to(rgbs.device)
            traj_mask = torch.full(
                [1, interpolation_length - 1], False
            ).to(rgbs.device)
            output["trajectory"] = self._policy(
                fake_traj,
                traj_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[..., :7],
                run_inference=True
            )
        else:
            print('Predict Keypose')
            pred = self._policy(
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
            )
            # Hackish, assume self._policy is an instance of Act3D
            output["action"] = self._policy.prepare_action(pred)

        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device


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


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        fine_sampling_ball_diameter=None,
        collision_checking=False,
        tasks_type='atomic',
        tasks=None,
        vlm_agent=None,
        tmp_dir=""
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )

        if tasks is not None:
            self.tasks_type = tasks_type
            task_classes = []
            task_class_variation_idx = []
            task_class_base = []
            for task in tasks:
                task_class_base.append('_'.join(task.split('_')[:-1]))
                # if task_class_base[-1] not in task_files:
                #     raise ValueError('Task %s not recognised!.' % task)
                if tasks_type == 'atomic':
                    task_class = name_to_class(task_class_base[-1], ATOMIC_TASKS_PY_FOLDER)
                elif tasks_type == 'compositional':
                    task_class = name_to_class(task_class_base[-1], COMPOSITIONAL_TASKS_PY_FOLDER)
                else:
                    task_class = name_to_class(task_class_base[-1], TASKS_PY_FOLDER) # task_file_to_task_class(task_class_base)
                task_class_variation_idx.append(int(task.split('_')[-1]))
                task_classes.append(task_class)
            
            self.env = MultiTaskRLBenchEnv(
                task_classes=task_classes,
                observation_config=self.obs_config,
                action_mode=self.action_mode,
                dataset_root=str(data_path),
                headless=headless,
                swap_task_every=25,
                base_cfg_name=task_class_base,
                task_class_variation_idx=task_class_variation_idx,
            )
        self.image_size = image_size

        self.vlm_agent: MultiModalAgent = vlm_agent
        self.tmp_dir = tmp_dir

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
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

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    def _get_home_pose(self):
        home_pose = np.array(Dummy("Panda_tip").get_pose())
        return np.concatenate((home_pose, [1, 0]))

    def _save_images(self, obs, base_target_cameras=['front', 'wrist'], step:int=0):
        # save image for vlm input
        base_images = []
        for target_cam in base_target_cameras:
            img_np = getattr(obs, f"{target_cam}_rgb")  # numpy array, uint8, shape (H, W, 3)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (3, H, W), float in [0,1]
            base_images.append(img_tensor)
        base_path: Path = Path(self.tmp_dir) / "images" / f"{step}"
        base_frame_images = []
        base_path.mkdir(parents=True, exist_ok=True)
        for j, img in enumerate(base_images):
            torchvision.utils.save_image(img, base_path / f"{j}.png")
            base_frame_images.append(str(base_path / f"{j}.png"))
        return base_frame_images

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        eval_demo_seed: int,
        max_steps: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        interpolation_length=50,
        num_history=0,
        eval_mode:str = "vanilla",
    ):
        device = actioner.device

        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)

        # descriptions, obs = task.reset()
        instruction, obs = self.env.reset_to_demo(eval_demo_seed)
        # instr
        # ============================================================================
        instruction_vanilla = instruction["vanilla"][0]
        instruction_oracle_half = instruction["oracle_half"][0].split("\n")
        instruction = ""
        instr_index = 0
        grasped_objects = self.env._rlbench_env._scene.robot.gripper.get_grasped_objects()  # noqa: SLF001
        prev_grasped_objects_len = len(grasped_objects)
        prev_gripper_state = 1.0 
        home_pose = self._get_home_pose()
        history_instr = []
        # ============================================================================

        move = Mover(self.env._task, max_tries=max_tries)
        reward = 0.0
        max_reward = 0.0

        for step_id in range(max_steps):
            
            # Fetch the current observation, and predict one action
            rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
            rgb = rgb.to(device)
            pcd = pcd.to(device)
            gripper = gripper.to(device)

            rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
            pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
            grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

            # Prepare proprioception history
            rgbs_input = rgbs[:, -1:][:, :, :, :3]
            pcds_input = pcds[:, -1:]
            if num_history < 1:
                gripper_input = grippers[:, -1]
            else:
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )

            # ============================================================================
            if eval_mode == "vanilla":
                instruction = instruction_vanilla
            elif eval_mode == "half":
                instruction = instruction_oracle_half[instr_index]
            elif eval_mode == "vlm":
                base_frame_images = self._save_images(obs, step=step_id)
                task_description = instruction_vanilla
                variables={
                    "image_paths": base_frame_images,
                    "task_description": task_description,
                }
                
                # TODO: history may contain current sub-task because it may not be performed yet
                variables["pre_sub_tasks"] = history_instr

                result = self.vlm_agent.run(
                    task_name="next_sub_task_w_history",
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
                except:
                    sub_task = ""
                instruction = sub_task
            else:
                raise ValueError(f"unknown eval mode: {eval_mode}")
            if verbose:
                print(f"instruction: {instruction}")
            actioner.load_episode(task_str, instruction)
            # ============================================================================

            output = actioner.predict(
                rgbs_input,
                pcds_input,
                gripper_input,
                interpolation_length=interpolation_length
            )

            if verbose:
                print(f"Step {step_id}")

            terminate = True

            # Update the observation based on the predicted action
            try:
                # Execute entire predicted trajectory step by step
                if output.get("trajectory", None) is not None:
                    trajectory = output["trajectory"][-1].cpu().numpy()
                    trajectory[:, -1] = trajectory[:, -1].round()

                    # execute
                    for action in trajectory:
                        #try:
                        #    collision_checking = self._collision_checking(task_str, step_id)
                        #    obs, reward, terminate, _ = move(action_np, collision_checking=collision_checking)
                        #except:
                        #    terminate = True
                        #    pass
                        collision_checking = self._collision_checking(task_str, step_id)
                        obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                # Or plan to reach next predicted keypoint
                else:
                    print("Plan with RRT")
                    action = output["action"]
                    action[..., -1] = torch.round(action[..., -1])
                    action = action[-1].detach().cpu().numpy()

                    collision_checking = self._collision_checking(task_str, step_id)
                    obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                max_reward = max(max_reward, reward)
                if reward == 1:
                    break
                
                if terminate:
                    print("The episode has terminated!")

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(task_str, eval_demo_seed, step_id, e)
                reward = 0
                #break
                
            # plan forward according to current task
            # ============================================================================
            grasped_objects = self.env._rlbench_env._scene.robot.gripper.get_grasped_objects()  # noqa: SLF001
            cur_grasped_objects_len = len(grasped_objects)
            cur_gripper_state = obs.gripper_open
            if (cur_grasped_objects_len != prev_grasped_objects_len or 
                    abs(cur_gripper_state - prev_gripper_state) > 1e-6):
                instr_index = (instr_index + 1) % len(instruction_oracle_half)
                # reset when grasped changed and gripper open
                if abs(obs.gripper_open - 1.000) < 1e-9:
                    obs, _, _ = self.env._task.step(home_pose)  # noqa: SLF001
            prev_grasped_objects_len = cur_grasped_objects_len
            prev_gripper_state = obs.gripper_open
            # ============================================================================

        success = True if reward == 1 else False
        # print(
        #     task_str,
        #     "Demo",
        #     eval_demo_seed,
        #     "max_reward",
        #     f"{reward:.2f}",
        #     "success",
        #     success
        # )
        
        vid = self.env._append_final_frame(success)
        return reward, vid

    def _collision_checking(self, task_str, step_id):
        """Collision checking for planner."""
        # collision_checking = True
        collision_checking = False
        # if task_str == 'close_door':
        #     collision_checking = True
        # if task_str == 'open_fridge' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'open_oven' and step_id == 3:
        #     collision_checking = True
        # if task_str == 'hang_frame_on_hanger' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'take_frame_off_hanger' and step_id == 0:
        #     for i in range(300):
        #         self.env._scene.step()
        #     collision_checking = True
        # if task_str == 'put_books_on_bookshelf' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'slide_cabinet_open_and_place_cups' and step_id == 0:
        #     collision_checking = True
        return collision_checking

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
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
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


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
