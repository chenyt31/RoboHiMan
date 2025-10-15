import random
import os
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, List

import tap
import torch
import numpy as np
import einops
import blosc
from tqdm import tqdm

from rlbench.demo import Demo
from src.utils.env_engine import RLBenchEnv
from src.utils.env_utils import keypoint_discovery, obs_to_attn, transform
from peract_colab.rlbench.utils import get_stored_demo
from src.utils.dataset_utils import Resize


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "train"
    seed: int = 2
    cameras: Tuple[str, ...] = ("front", "left_shoulder", "right_shoulder", "wrist")
    max_episodes_per_task: int = 200
    image_size: str = "256,256"
    offset: int = 0
    num_workers: int = 0
    frame_interval: int = 10
    training: bool = True
    image_rescale: str = "1.0,1.0"

def gripper_change(demo, i, threshold=2):    
    start = max(0, i - threshold)
    for k in range(start, i):
        if demo[k].gripper_open != demo[i].gripper_open:
            return True
    return False


def touch_change(demo, i, threshold=4, delta=0.005):
    start = max(0, i - threshold)
    for k in range(start, i):
        if np.allclose(demo[k].gripper_touch_forces, 0, atol=delta) != \
           np.allclose(demo[i].gripper_touch_forces, 0, atol=delta):
            return True
    return False


def get_attn_indices_from_demo(demo: Demo, cameras: Tuple[str, ...]) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)
    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]


def get_observation(data_path: Path, index: int, env: RLBenchEnv, frame_interval: int):
    demo = get_stored_demo(data_path=data_path, index=index)
    num_frames = len(demo)

    key_frame = keypoint_discovery(demo)
    key_frame.insert(0, 0)

    goal_keyframe = []
    for i in range(len(key_frame) - 1):
        if gripper_change(demo, key_frame[i]) or touch_change(demo, key_frame[i]):
            goal_keyframe.append(key_frame[i])
    goal_keyframe.append(key_frame[-1])

    sampled_frames = list(range(0, num_frames, frame_interval))

    state_ls = []
    gripper_pose_ls = []
    action_ls = []
    subgoal_action_ls = []
    subgoal_index_ls = []
    goal_rgb_ls = []

    for i in sampled_frames:
        obs = demo[i]
        state, gripper_pose = env.get_obs_action(obs)
        state = transform(state)
        state_ls.append(state.unsqueeze(0))
        gripper_pose_ls.append(gripper_pose)

        next_kf = next((kf for kf in key_frame if kf > i), key_frame[-1])
        _, action = env.get_obs_action(demo[next_kf])
        action_ls.append(action)

        goal_index = next((idx for idx, gk in enumerate(goal_keyframe) if gk > i), len(goal_keyframe) - 1)
        _, goal_action = env.get_obs_action(demo[goal_keyframe[goal_index]])
        subgoal_action_ls.append(goal_action)
        subgoal_index_ls.append(goal_index)

        # === 取 goal_rgb ===
        goal_obs = demo[goal_keyframe[goal_index]]
        goal_state, _ = env.get_obs_action(goal_obs)
        goal_state = transform(goal_state)
        goal_rgb_ls.append(goal_state.unsqueeze(0))  # 跟 state_ls 对齐

    return demo, state_ls, gripper_pose_ls, action_ls, subgoal_action_ls, subgoal_index_ls, goal_rgb_ls


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments):
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        self.items = []
        tasks = os.listdir(args.data_dir)
        for task_str in tasks:
            for eps_idx in range(args.offset, args.max_episodes_per_task):
                episodes_dir = args.data_dir / task_str / f"all_variations/episodes/episode{eps_idx}"
                if episodes_dir.exists():
                    self.items.append((task_str, args.data_dir / task_str / f"all_variations/episodes", eps_idx))
        print(f"Found {len(self.items)} episodes")
        self.num_items = len(self.items)
        self.args = args

        # If training, initialize augmentation classes
        self._training = args.training
        if self._training:
            self._resize = Resize(scales=tuple(
                float(x) for x in args.image_rescale.split(",")
            ))

    def __len__(self) -> int:
        return self.num_items

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def __getitem__(self, index: int) -> dict:
        task_str, episode_dir, eps_idx = self.items[index]

        (demo, state_ls, gripper_pose_ls, action_ls,
        subgoal_action_ls, subgoal_index_ls, goal_rgb_ls) = get_observation(
            episode_dir, eps_idx, self.env, frame_interval=self.args.frame_interval
        )

        frame_ids = list(range(len(state_ls)))
        action = torch.stack(action_ls)
        gripper_pose = torch.stack(gripper_pose_ls)
        sub_goal = torch.stack(subgoal_action_ls)

        state_tensor = einops.rearrange(
            state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(self.args.cameras),
            m=2,
        )

        goal_tensor = einops.rearrange(
            goal_rgb_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(self.args.cameras),
            m=2,
        )

        lang_path = episode_dir / f"episode{eps_idx}" / "variation_descriptions.pkl"
        with open(lang_path, "rb") as f:
            lang_dict = pickle.load(f)

        rgbs = state_tensor[:, :, 0]
        pcds = state_tensor[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        goal_rgbs = goal_tensor[:, :, 0]
        goal_rgbs = self._unnormalize_rgb(goal_rgbs)

        # Augmentations
        if self._training:
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

            modals_goal = self._resize(rgbs=goal_rgbs)
            goal_rgbs = modals_goal["rgbs"]

        instr_vanilla = lang_dict["vanilla"][0]
        instr_oracle_half = lang_dict["oracle_half"][0].split('\n')

        ret_dict = {
            "task_str": [task_str for _ in frame_ids],
            "rgbs": rgbs,
            "pcds": pcds,
            "action": action,
            "gripper_pose": gripper_pose,
            "instr_vanilla": [instr_vanilla for _ in frame_ids],
            "instr_oracle_half": [
                instr_oracle_half[subgoal_index_ls[i] % (len(instr_oracle_half))]
                for i in frame_ids
            ],
            "sub_goal": sub_goal,
            "goal_rgbs": goal_rgbs,
        }

        return ret_dict

if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )
    # 调试一个 batch
    for i, ret_dict in enumerate(tqdm(dataloader, total=len(dataset))):
        print(f"Sample {i} ====================")
        ret_dict = ret_dict[0]

        print("task_str:", ret_dict["task_str"])
        print("rgbs shape:", ret_dict["rgbs"].shape)
        print("pcds shape:", ret_dict["pcds"].shape)
        print("action shape:", ret_dict["action"].shape)
        print("gripper_pose shape:", ret_dict["gripper_pose"].shape)
        print("instr_vanilla:", ret_dict["instr_vanilla"])
        print("instr_oracle_half:", ret_dict["instr_oracle_half"])
        print("sub_goal shape:", ret_dict["sub_goal"].shape)
        break  # 只查看第一个样本