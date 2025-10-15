from collections import defaultdict
import json
import pickle

from tqdm import tqdm
import numpy as np
from typing import Tuple
from pathlib import Path
import tap
from numpy import random
from src.utils.data_engine import Dataset
import torch
import torchvision
from src.high_level.vlm.agent import MultiModalAgent
from src.high_level.vlm.models.qwen_vl_runner import QwenVLRunner
from jinja2 import Template

class Arguments(tap.Tap):

    # Training and validation datasets
    cameras: Tuple[str, ...] = ("front", "wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 200
    data_dir: Path = Path("HiMan_data/train") # modify
    offset: int = 0
    num_workers: int = 0
    batch_size: int = 1
    frame_interval: int = 10

    # Data augmentations
    training: bool = True
    image_rescale: str = "1.0,1.0"  # (min, max), "1.0,1.0" for no rescaling

    # Data saving
    save_dir: Path = Path("HiMan_data/train_L1_L2_L3_L4_package_vlm_planning") # modify
    seed: int = 2
    
    # mode
    multi_view_enabled: bool = False
    history_enabled: bool = False

    # cot
    gen_cot_vlm_ckpt_dir: str = "Qwen/Qwen2.5-VL-7B-Instruct" # pretrained_model_name_or_path
    cot_vlm_device: str = "cuda:0"
    prompt_dir: Path = Path("src/high_level/prompts") # modify
    prompt_model_name: str = "qwen_vl"
    
    # checkpoint settings
    checkpoint_interval: int = 10  # Save checkpoint every N episodes
    checkpoint_file: str = "checkpoint.pkl"  # Checkpoint filename


def get_target_images(rgbs: torch.Tensor, cameras: Tuple[str, ...], target_cams: Tuple[str, ...]):
    """
    Get target images from rgbs.

    Args:
        rgbs: (n_cam, 3, H, W)
        cameras: Tuple[str, ...]
        target_cams: Tuple[str, ...]

    Returns:
        img_tensors: (n_cam, 3, H, W)
    """
    img_tensors = []
    for cam_idx, cam_name in enumerate(cameras):
        if cam_name not in target_cams:
            continue
        img_tensors.append(rgbs[cam_idx])
    return img_tensors


def save_checkpoint(checkpoint_path: Path, base_json_list: list, task_dict: dict, 
                   processed_episodes: set, total_episodes: int):
    """Save checkpoint with current progress"""
    checkpoint_data = {
        'base_json_list': base_json_list,
        'task_dict': dict(task_dict),
        'processed_episodes': processed_episodes,
        'total_episodes': total_episodes
    }
    
    # Save to temporary file first, then rename to avoid corruption
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Atomic rename
    temp_path.rename(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint if exists"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"Resuming from episode {len(checkpoint_data['processed_episodes'])}/{checkpoint_data['total_episodes']}")
            return checkpoint_data
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    return None


if __name__ == '__main__':
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
        shuffle=False
    )

    # Checkpoint paths
    checkpoint_path = args.save_dir / args.checkpoint_file
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        base_json_list = checkpoint_data['base_json_list']
        task_dict = defaultdict(int, checkpoint_data['task_dict'])
        processed_episodes = checkpoint_data['processed_episodes']
        total_episodes = checkpoint_data['total_episodes']
        print(f"Resuming from checkpoint with {len(base_json_list)} processed samples")
    else:
        base_json_list = []
        task_dict = defaultdict(int)
        processed_episodes = set()
        total_episodes = len(dataloader)

    # views
    if args.multi_view_enabled: # base camera
        base_target_cameras: Tuple[str, ...] = ("front", "wrist", "left_shoulder", "right_shoulder")
    else:
        base_target_cameras: Tuple[str, ...] = ("front", "wrist")
    cot_target_cameras: Tuple[str, ...] = ("front", "wrist") # cot camera

    # load cot vlm
    cot_vlm_runner = QwenVLRunner(mode="local", model_name=args.gen_cot_vlm_ckpt_dir, device=args.cot_vlm_device)
    cot_agent = MultiModalAgent(cot_vlm_runner, base_dir=args.prompt_dir, prompt_model_name=args.prompt_model_name)

    # load template
    if not args.history_enabled:
        input_template_file = args.prompt_dir / "data" / "next_sub_task_input.jinja2"
    else:
        input_template_file = args.prompt_dir / "data" / "next_sub_task_w_history_input.jinja2"
    output_template_file = args.prompt_dir / "data" / "next_sub_task_output.jinja2"

    input_template: Template = Template(input_template_file.read_text())
    output_template: Template = Template(output_template_file.read_text())

    for episode_idx, sample in enumerate(tqdm(dataloader, desc="Processing data", initial=len(processed_episodes))):  # per episode
        sample = sample[0]
        
        # Skip if already processed
        if episode_idx in processed_episodes:
            continue

        rgbs = sample['rgbs']              # (n_frame, n_cam, 3, H, W)
        goal_rgbs = sample['goal_rgbs']    # (n_frame, n_cam, 3, H, W)

        tasks = sample['task_str']
        instr_vanilla = sample['instr_vanilla']
        instr_oracle_half = sample["instr_oracle_half"]

        for i in range(rgbs.shape[0]):
            # task description
            task_description = instr_vanilla[i]

            # images
            base_images = get_target_images(rgbs[i], args.cameras, base_target_cameras)
            start_images = get_target_images(rgbs[i], args.cameras, cot_target_cameras)
            end_images = get_target_images(goal_rgbs[i], args.cameras, cot_target_cameras)

            # save images
            base_path: Path = args.save_dir / "images" / tasks[i] / f"{task_dict[tasks[i]]}" # base vlm
            base_goal_path: Path = args.save_dir / "images_goal" / tasks[i] / f"{task_dict[tasks[i]]}" # cot vlm
            base_frame_images = []
            start_frame_images = []
            end_frame_images = []

            # Create directories
            base_path.mkdir(parents=True, exist_ok=True)
            base_goal_path.mkdir(parents=True, exist_ok=True)

            for j, img in enumerate(base_images):
                torchvision.utils.save_image(img, base_path / f"{j}.png")
                base_frame_images.append(base_path / f"{j}.png")

            for j, img in enumerate(start_images):
                torchvision.utils.save_image(img, base_goal_path / f"start_{j}.png")
                start_frame_images.append(base_goal_path / f"start_{j}.png")

            for j, img in enumerate(end_images):
                torchvision.utils.save_image(img, base_goal_path / f"end_{j}.png")
                end_frame_images.append(base_goal_path / f"end_{j}.png")

            # cot reasoning
            variables={
                "task_description": task_description,
                "start_frame_images": start_frame_images,
                "end_frame_images": end_frame_images,
                "sub_task": instr_oracle_half[i]
            }
            
            history_index = min(i + 1, len(instr_oracle_half) - 1) \
                if random.random() < 0.2 else i # history may contain current sub-task because it may not be performed yet
            
            if args.history_enabled:
                variables["pre_sub_tasks"] = instr_oracle_half[:history_index]
            try:
                cot_reasoning = cot_agent.run(
                    task_name="gen_cot" if not args.history_enabled else "gen_cot_w_history",
                    variables=variables,
                    fields=("reasoning",)
                    )[0]['reasoning'][0]
            except Exception as e:
                print(f"Error in cot reasoning: {e}")
                continue

            # input template
            input_template_rendered = input_template.render(
                task_description=task_description,
                image_paths=base_frame_images,
                pre_sub_tasks=instr_oracle_half[:history_index] if args.history_enabled else None
            )

            # output template
            output_template_rendered = output_template.render(
                reasoning=cot_reasoning,
                sub_task=instr_oracle_half[i]
            )

            base_json_list.append({
                "messages": [
                    {
                        "content": "You are a helpful assistant.",
                        "role": "system"
                    },
                    {
                        "content": input_template_rendered,
                        "role": "user"
                    },
                    {
                        "content": output_template_rendered,
                        "role": "assistant"
                    },
                ],
                "images": [str(p) for p in base_frame_images]
            })

            task_dict[tasks[i]] += 1

        # Mark episode as processed
        processed_episodes.add(episode_idx)
        
        # Save checkpoint periodically
        if (episode_idx + 1) % args.checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, base_json_list, task_dict, processed_episodes, total_episodes)
        
    # Final checkpoint save
    save_checkpoint(checkpoint_path, base_json_list, task_dict, processed_episodes, total_episodes)

    # save data info
    with open(args.save_dir / (args.save_dir.name + ".json"), "w") as f:
        json.dump(base_json_list, f)

    with (args.save_dir / "dataset_info.json").open("w") as f:
        json.dump({
            args.save_dir.name: {
                "file_name": args.save_dir.name + ".json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages",
                    "images": "images"
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system"
                }
            }
        }, f)
    
    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Processing completed successfully. Checkpoint file removed.")