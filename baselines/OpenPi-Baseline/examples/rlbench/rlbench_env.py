from functools import reduce
import json
from pathlib import Path
from typing import Any

from colosseum import ASSETS_ATOMIC_CONFIGS_FOLDER
from colosseum import ASSETS_ATOMIC_JSON_FOLDER
from colosseum import ASSETS_COMPOSITIONAL_CONFIGS_FOLDER
from colosseum import ASSETS_COMPOSITIONAL_JSON_FOLDER
from colosseum import ATOMIC_TASKS_PY_FOLDER
from colosseum import ATOMIC_TASKS_TTM_FOLDER
from colosseum import COMPOSITIONAL_TASKS_PY_FOLDER
from colosseum import COMPOSITIONAL_TASKS_TTM_FOLDER
from colosseum import TASKS_PY_FOLDER
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import name_to_class
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError
from pyrep.errors import IKError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.task import Task
from rlbench.observation_config import ObservationConfig
import torch

from examples.rlbench.rlbench_utils import Actioner
from examples.rlbench.rlbench_utils import create_obs_config


class MoveJointArmThenGripper(ActionMode):
    """The arm action is first applied, followed by the gripper action."""

    def action(self, scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size : arm_act_size + 1])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(self.gripper_action_mode.action_shape(scene))


def change_case(str):
    return reduce(lambda x, y: x + ("_" if y.isupper() else "") + y, str).lower()


def get_spreadsheet_config(base_cfg: DictConfig, collection_cfg: dict[str, Any], spreadsheet_idx: int) -> DictConfig:
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

    collections_variation_cfg = collection_cfg["strategy"][spreadsheet_idx]["variations"]
    for collection_var_cfg in collections_variation_cfg:
        var_type = collection_var_cfg["type"]
        var_name = collection_var_cfg["name"]
        var_enabled = collection_var_cfg["enabled"]
        for variation_cfg in spreadsheet_cfg.env.scene.factors:
            if variation_cfg.variation != var_type:
                continue
            if var_name == "any" or ("name" in variation_cfg and variation_cfg.name == var_name):
                variation_cfg.enabled = var_enabled

    return spreadsheet_cfg


class MultiTaskRLBenchEnv:
    def __init__(
        self,
        task_classes: list[type[Task]],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        dataset_root: str = "",
        swap_task_every: int = 1,
        base_cfg_name=None,
        task_class_variation_idx=None,
        *,
        headless: bool = True,
    ):
        # super(MultiTaskRLBenchEnv, self).__init__()

        self._task_classes = task_classes
        self._observation_config = observation_config
        # self._rlbench_env = Environment(
        #     action_mode=action_mode, obs_config=observation_config,
        #     dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ""
        self._lang_goal = "unknown goal"
        self._swap_task_every = swap_task_every

        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__): i for i, tc in enumerate(self._task_classes)}
        self._base_cfg_name = base_cfg_name
        self._task_class_variation_idx = task_class_variation_idx
        self._action_mode = action_mode
        self._observation_config = observation_config
        self._dataset_root = dataset_root
        self._headless = headless

        self._record_cam = None
        self._recorded_images = []

    def _set_new_task(self, *, shuffle=False):
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
        self._lang_goal = (
            descriptions.get("vanilla", [descriptions[0]])[0] if isinstance(descriptions, dict) else descriptions[0]
        )
        # self._lang_goal = descriptions[0] # first description variant

    def launch(self, task_type=None):
        if task_type == "atomic":
            assets_configs_folder = Path(ASSETS_ATOMIC_CONFIGS_FOLDER)
            assets_json_folder = Path(ASSETS_ATOMIC_JSON_FOLDER)
            tasks_ttm_folder = Path(ATOMIC_TASKS_TTM_FOLDER)
        elif task_type == "compositional":
            assets_configs_folder = Path(ASSETS_COMPOSITIONAL_CONFIGS_FOLDER)
            assets_json_folder = Path(ASSETS_COMPOSITIONAL_JSON_FOLDER)
            tasks_ttm_folder = Path(COMPOSITIONAL_TASKS_TTM_FOLDER)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        base_cfg_path = assets_configs_folder / f"{self._base_cfg_name[self._active_task_id]}.yaml"
        if base_cfg_path.exists():
            with base_cfg_path.open("r") as f:
                base_cfg = OmegaConf.load(f)
        else:
            raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

        collection_cfg_path: Path = assets_json_folder / f"{base_cfg.env.task_name}.json"
        with collection_cfg_path.open("r") as fh:
            collection_cfg: Any | None = json.load(fh)

        if not collection_cfg or "strategy" not in collection_cfg:
            return 1

        # num_spreadsheet_idx = len(collection_cfg["strategy"])

        if self._task_class_variation_idx is not None:
            full_config = get_spreadsheet_config(
                base_cfg,
                collection_cfg,
                self._task_class_variation_idx[self._active_task_id],
            )
            _, env_cfg = full_config.data, full_config.env
        else:
            env_cfg = None

        self._rlbench_env = EnvironmentExt(
            action_mode=self._action_mode,
            obs_config=self._observation_config,
            path_task_ttms=tasks_ttm_folder,
            dataset_root=self._dataset_root,
            headless=self._headless,
            env_config=env_cfg,
        )

        self._rlbench_env.launch()
        self._set_new_task()

        # record
        self._task._scene.register_step_callback(self._my_callback)  # noqa: SLF001
        cam_placeholder = Dummy("cam_cinematic_placeholder")
        cam_base = Dummy("cam_cinematic_base")
        cam_base.rotate([0, 0, np.pi * 0.75])
        self._record_cam = VisionSensor.create([320, 180])
        self._record_cam.set_explicit_handling(True)
        self._record_cam.set_pose(cam_placeholder.get_pose())
        self._record_cam.set_render_mode(RenderMode.OPENGL)

        return 0

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
        if isinstance(descriptions, dict) and "vanilla" in descriptions:
            self._lang_goal = descriptions["vanilla"][0]
        else:
            self._lang_goal = descriptions[0]
        # self._lang_goal = descriptions[0] # first description variant

        return descriptions, obs

    def append_final_frame(self, *, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10,) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

        return np.array(self._recorded_images).transpose((0, 3, 1, 2))

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0

        if self._task_class_variation_idx is not None:
            self._task.set_variation(-1)
            self._task._task.task_path = (  # noqa: SLF001
                self._task._task.name + f"_{self._task_class_variation_idx[self._active_task_id]!s}"  # noqa: SLF001
            )
        else:
            self._task.set_variation(-1)

        d = self._task.get_demos(1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        self._recorded_images.clear()

        descriptions, obs = self._task.reset_to_demo(d)
        self.descriptions = descriptions
        if isinstance(descriptions, dict) and "vanilla" in descriptions:
            self._lang_goal = descriptions["vanilla"][0]
        else:
            self._lang_goal = descriptions[0]
        return descriptions, obs


class RLBenchEnv:
    def __init__(
        self,
        data_path,
        image_size=(256, 256),
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        tasks=None,
        *,
        tasks_type="atomic",
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_mask=False,
        headless=True,
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
            apply_cameras=self.apply_cameras,
        )

        self.action_mode = MoveJointArmThenGripper(arm_action_mode=JointPosition(), gripper_action_mode=Discrete())
        self.tasks_type = tasks_type
        task_classes = []
        task_class_variation_idx = []
        task_class_base = []
        for task in tasks:
            task_class_base.append("_".join(task.split("_")[:-1]))
            # if task_class_base[-1] not in task_files:
            #     raise ValueError('Task %s not recognised!.' % task)
            if tasks_type == "atomic":
                task_class = name_to_class(task_class_base[-1], ATOMIC_TASKS_PY_FOLDER)
            elif tasks_type == "compositional":
                task_class = name_to_class(task_class_base[-1], COMPOSITIONAL_TASKS_PY_FOLDER)
            else:
                task_class = name_to_class(
                    task_class_base[-1], TASKS_PY_FOLDER
                )  # task_file_to_task_class(task_class_base)
            task_class_variation_idx.append(int(task.split("_")[-1]))
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

    def _get_home_pose(self):
        home_pose = np.array(Dummy("Panda_tip").get_pose())
        return np.concatenate((home_pose, [1, 0]))

    @torch.no_grad()
    def evaluate_task_on_one_variation(
        self,
        task_str: str,
        eval_demo_seed: int,
        max_steps: int,
        actioner: Actioner,
        eval_mode: str = "half",  # vanilla | half | vlm
        *,
        verbose: bool = False,
    ):
        # Reset task to demo state
        try:
            instruction, obs = self.env.reset_to_demo(eval_demo_seed)
        except Exception as e:
            return None, None
        # instr
        # ============================================================================
        instruction_vanilla = instruction["vanilla"][0]
        instruction_oracle_half = instruction["oracle_half"][0].split("\n")
        instruction = ""
        instr_index = 0
        grasped_objects = self.env._rlbench_env._scene.robot.gripper.get_grasped_objects()  # noqa: SLF001
        prev_grasped_objects_len = len(grasped_objects)
        prev_gripper_state = 1.0
        # ============================================================================

        reward = 0.0

        # Add task information to kwargs
        kwargs = {}
        kwargs["task_str"] = task_str

        actioner.reset()
        home_pose = self._get_home_pose()

        for step_id in range(max_steps):
            # ============================================================================
            if eval_mode == "vanilla":
                instruction = instruction_vanilla
            elif eval_mode == "half":
                instruction = instruction_oracle_half[instr_index]
            elif eval_mode == "vlm":
                pass
            else:
                raise ValueError(f"unknown eval mode: {eval_mode}")
            if verbose:
                print(f"instruction: {instruction}")
            # ============================================================================

            # Get front RGB image
            front_rgb = obs.front_rgb
            wrist_rgb = obs.wrist_rgb
            proprio = np.concatenate(
                [obs.joint_positions, [obs.gripper_open]],
                axis=-1,
                dtype=np.float32,
            )

            # Get action prediction
            trajectory = actioner.predict(proprio, front_rgb, wrist_rgb, instruction, **kwargs)

            try:
                # Execute actions
                for action in trajectory:
                    obs, reward, terminate = self.env._task.step(action)  # noqa: SLF001

                if reward == 1:
                    break

                if terminate:
                    print("Episode terminated!")
                    break

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(task_str, eval_demo_seed, step_id, e)
                reward = 0
                # break

            # plan forward according to current task
            # ============================================================================
            grasped_objects = self.env._rlbench_env._scene.robot.gripper.get_grasped_objects()  # noqa: SLF001
            cur_grasped_objects_len = len(grasped_objects)
            if cur_grasped_objects_len != prev_grasped_objects_len or abs(obs.gripper_open - prev_gripper_state) > 1e-6:
                instr_index = (instr_index + 1) % len(instruction_oracle_half)
                actioner.reset()

                # reset when grasped changed and gripper open
                if abs(obs.gripper_open - 1.000) < 1e-9:
                    obs, reward, terminate = self.env._task.step(home_pose)  # noqa: SLF001
            prev_grasped_objects_len = cur_grasped_objects_len
            prev_gripper_state = obs.gripper_open
            # ============================================================================

        success = reward == 1
        return reward, self.env.append_final_frame(success=success)
