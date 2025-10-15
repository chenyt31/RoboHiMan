import os
import random
import itertools
from typing import Tuple, Dict, List
import pickle
from pathlib import Path
import json

import blosc
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo

from utils.utils_with_rlbench import (
    RLBenchEnv,
    keypoint_discovery,
    obs_to_attn,
    transform,
)
# Adapted from https://github.com/stepjam/RLBench/blob/master/rlbench/utils.py

import os
import pickle
import numpy as np
from PIL import Image

from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor

import json
from pathlib import Path
import itertools
from typing import List, Tuple, Literal, Dict

import transformers
from tqdm.auto import tqdm

TextEncoder = Literal["bert", "clip"]

# constants
EPISODE_FOLDER = 'episode%d'

CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_TYPES = [IMAGE_RGB, IMAGE_DEPTH]
IMAGE_FORMAT  = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

DEPTH_SCALE = 2**24 - 1

# functions
def get_stored_demo(data_path, index):
  episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
  
  # low dim pickle file
  with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
    obs = pickle.load(f)

  # variation number
  with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
    obs.variation_number = pickle.load(f)

  num_steps = len(obs)
  for i in range(num_steps):
    obs[i].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
    obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
    obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
    obs[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

    obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
    far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
    obs[i].front_depth = near + obs[i].front_depth * (far - near)

    obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
    far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
    obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)

    obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
    far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
    obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)

    obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
    far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
    obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)

    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth, 
                                                                                    obs[i].misc['front_camera_extrinsics'],
                                                                                    obs[i].misc['front_camera_intrinsics'])
    obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].left_shoulder_depth, 
                                                                                            obs[i].misc['left_shoulder_camera_extrinsics'],
                                                                                            obs[i].misc['left_shoulder_camera_intrinsics'])
    obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].right_shoulder_depth, 
                                                                                             obs[i].misc['right_shoulder_camera_extrinsics'],
                                                                                             obs[i].misc['right_shoulder_camera_intrinsics'])
    obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].wrist_depth, 
                                                                                           obs[i].misc['wrist_camera_extrinsics'],
                                                                                           obs[i].misc['wrist_camera_intrinsics'])
    
  return obs

def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer

class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "c2farm"
    seed: int = 2
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")
    image_size: str = "256,256"
    output: Path = Path(__file__).parent / "datasets"
    max_episodes_per_task: int = 100
    offset: int = 0
    num_workers: int = 0
    store_intermediate_actions: int = 1

    # instructions
    encoder: TextEncoder = "clip"
    model_max_length: int = 53
    device: str = "cuda"

    # mode
    train_mode: str = "half"


def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]


def get_observation(data_path: Path, index: int, env: RLBenchEnv,
                    store_intermediate_actions: bool):
    demo = get_stored_demo(data_path=data_path, index=index)

    key_frame = keypoint_discovery(demo)
    key_frame.insert(0, 0)

    keyframe_state_ls = []
    keyframe_action_ls = []
    intermediate_action_ls = []

    for i in range(len(key_frame)):
        state, action = env.get_obs_action(demo[key_frame[i]])
        state = transform(state)
        keyframe_state_ls.append(state.unsqueeze(0))
        keyframe_action_ls.append(action.unsqueeze(0))

        if store_intermediate_actions and i < len(key_frame) - 1:
            intermediate_actions = []
            for j in range(key_frame[i], key_frame[i + 1] + 1):
                _, action = env.get_obs_action(demo[j])
                intermediate_actions.append(action.unsqueeze(0))
            intermediate_action_ls.append(torch.cat(intermediate_actions))

    return demo, keyframe_state_ls, keyframe_action_ls, intermediate_action_ls


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: Arguments):
        # load RLBench environment
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

        self.train_mode = args.train_mode
        assert self.train_mode in ("vanilla", "half")

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task_str, episode_dir, eps_idx = self.items[index]
        taskvar_dir = args.output / f"{task_str}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        output_file = taskvar_dir / f"ep{eps_idx}.dat"
        if output_file.exists():
            print(f"File {output_file} already exists, skipping...")
            return None

        (demo,
         keyframe_state_ls,
         keyframe_action_ls,
         intermediate_action_ls) = get_observation(
            episode_dir, eps_idx, self.env,
            bool(args.store_intermediate_actions)
        )

        state_ls = einops.rearrange(
            keyframe_state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task_str, demo, args.cameras)

        lang_path = episode_dir / f"episode{eps_idx}" / "variation_descriptions.pkl"
        with open(lang_path, "rb") as f:
            lang_dict = pickle.load(f)

        state_dict: List = [[] for _ in range(7)]
        print("Demo {}".format(eps_idx))
        state_dict[0].extend(frame_ids)
        state_dict[1] = state_ls[:-1].numpy()
        state_dict[2].extend(keyframe_action_ls[1:])
        state_dict[3].extend(attn_indices)
        state_dict[4].extend(keyframe_action_ls[:-1])  # gripper pos
        state_dict[5].extend(intermediate_action_ls)   # traj from gripper pos to keyframe action

        tokenizer = load_tokenizer(args.encoder)
        tokenizer.model_max_length = args.model_max_length

        if self.train_mode != "half":
            descriptions = [lang_dict["vanilla"][0]] * len(frame_ids)
        else:
            descriptions_half = lang_dict["oracle_half"][0].split('\n')
            descriptions = []
            index = 0
            for i in range(num_frames):
                descriptions.append(descriptions_half[index])
                if abs(keyframe_action_ls[i][0][-1]-keyframe_action_ls[i+1][0][-1]) > 1e-9:
                    index = (index+1) % len(descriptions_half)

        model = load_model(args.encoder)
        model = model.to(args.device)
        tokens = tokenizer(descriptions, padding="max_length")["input_ids"]
        lengths = [len(t) for t in tokens]
        if any(l > args.model_max_length for l in lengths):
            raise RuntimeError(f"Too long instructions: {lengths}")

        tokens = torch.tensor(tokens).to(args.device)
        with torch.no_grad():
            pred = model(tokens).last_hidden_state.cpu()
        state_dict[6].extend(pred)

        with open(taskvar_dir / f"ep{eps_idx}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


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

    for _ in tqdm(dataloader):
        continue
