from collections import defaultdict
import json

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
    data_dir: Path = Path("HiMan_data/train")
    offset: int = 0
    num_workers: int = 0
    batch_size: int = 1
    frame_interval: int = 10

    # Data augmentations
    training: bool = True
    image_rescale: str = "1.0,1.0"  # (min, max), "1.0,1.0" for no rescaling

    # Data saving
    save_dir: Path = Path("HiMan_data/train_L1_L2_L3_L4_package_vlm_planning")
    seed: int = 2
    
    # mode
    multi_view_enabled: bool = False
    history_enabled: bool = False

    # cot
    vlm_ckpt_dir: str = "Qwen/Qwen2.5-VL-7B-Instruct" # pretrained_model_name_or_path
    vlm_device: str = "cuda:0"
    prompt_dir: Path = Path("src/high_level/prompts")
    prompt_model_name: str = "qwen_vl"
    
    # checkpoint settings
    checkpoint_interval: int = 2  # Save checkpoint every N episodes
    checkpoint_file: str = "checkpoint.json"  # Checkpoint filename (use JSON, not PKL)


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


def save_checkpoint(checkpoint_path: Path, records: list, task_dict: dict,
                   processed_episodes: set, total_episodes: int,
                   num_success: int, num_total: int):
    """Save checkpoint with current progress to JSON"""
    checkpoint_data = {
        'records': records,  # list of per-sample evaluation dicts
        'task_dict': dict(task_dict),
        'processed_episodes': list(processed_episodes),
        'total_episodes': total_episodes,
        'num_success': num_success,
        'num_total': num_total,
        'overall_success_rate': (num_success / num_total) if num_total > 0 else 0.0,
    }

    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    temp_path.rename(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint from JSON if exists"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"Resuming from episode {len(checkpoint_data.get('processed_episodes', []))}/{checkpoint_data.get('total_episodes', 0)}")
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
        records = checkpoint_data.get('records', [])
        task_dict = defaultdict(int, checkpoint_data.get('task_dict', {}))
        processed_episodes = set(checkpoint_data.get('processed_episodes', []))
        total_episodes = checkpoint_data.get('total_episodes', len(dataloader))
        num_success = int(checkpoint_data.get('num_success', 0))
        num_total = int(checkpoint_data.get('num_total', 0))
        print(f"Resuming from checkpoint with {len(records)} processed samples")
    else:
        records = []
        task_dict = defaultdict(int)
        processed_episodes = set()
        total_episodes = len(dataloader)
        num_success = 0
        num_total = 0

    # views
    if args.multi_view_enabled: # base camera
        base_target_cameras: Tuple[str, ...] = ("front", "wrist", "left_shoulder", "right_shoulder")
    else:
        base_target_cameras: Tuple[str, ...] = ("front", "wrist")

    # load vlm
    vlm_runner = QwenVLRunner(mode="local", model_name=args.vlm_ckpt_dir, device=args.vlm_device)
    agent = MultiModalAgent(vlm_runner, base_dir=args.prompt_dir, prompt_model_name=args.prompt_model_name)

    for episode_idx, sample in enumerate(tqdm(dataloader, desc="Processing data")):  # per episode
        sample = sample[0]
        
        # Skip if already processed
        if episode_idx in processed_episodes:
            continue

        rgbs = sample['rgbs']              # (n_frame, n_cam, 3, H, W)
        tasks = sample['task_str']
        instr_vanilla = sample['instr_vanilla']
        instr_oracle_half = sample["instr_oracle_half"]

        for i in range(rgbs.shape[0]):
            # task description
            task_description = instr_vanilla[i]
            gt_sub_task = instr_oracle_half[i]

            # images
            base_images = get_target_images(rgbs[i], args.cameras, base_target_cameras)
            base_path: Path = args.save_dir / "images" / tasks[i] / f"{task_dict[tasks[i]]}" # base vlm
            base_frame_images = []
            base_path.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(base_images):
                torchvision.utils.save_image(img, base_path / f"{j}.png")
                base_frame_images.append(base_path / f"{j}.png")

            variables={
                "image_paths": base_frame_images,
                "task_description": task_description,
            }
            
            history_index = min(i + 1, len(instr_oracle_half) - 1) \
                if random.random() < 0.2 else i # history may contain current sub-task because it may not be performed yet
            
            if args.history_enabled:
                variables["pre_sub_tasks"] = instr_oracle_half[:history_index]

            result = agent.run(
                task_name="next_sub_task" if not args.history_enabled else "next_sub_task_w_history",
                variables=variables,
                fields=("reasoning", "sub_task")
            )[0]
            try:
                reasoning = result['reasoning'][0]
            except Exception as e:
                print(f"Error generating COT reasoning: {str(e)}")
                reasoning = "" 
            try: 
                sub_task = result['sub_task'][0]
            except Exception as e:
                print(f"Error generating Sub task: {str(e)}")
                sub_task = "" 

            task_dict[tasks[i]] += 1

            # Evaluate prediction vs ground truth
            pred_norm = str(sub_task).strip().lower()
            gt_norm = str(gt_sub_task).strip().lower()
            success = (pred_norm == gt_norm)
            num_total += 1
            if success:
                num_success += 1

            running_acc = num_success / num_total if num_total > 0 else 0.0

            # Record per-sample evaluation
            record = {
                'episode_idx': episode_idx,
                'frame_idx': i,
                'task_name': tasks[i],
                'task_description': task_description,
                'reasoning': reasoning,
                'pred_sub_task': sub_task,
                'gt_sub_task': gt_sub_task,
                'success': success,
                'running_success_rate': running_acc,
            }
            records.append(record)

        # Mark episode as processed
        processed_episodes.add(episode_idx)
        
        # Save checkpoint periodically
        if (episode_idx + 1) % args.checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, records, task_dict, processed_episodes, total_episodes, num_success, num_total)
        

    # Final checkpoint save (keep the JSON for review)
    save_checkpoint(checkpoint_path, records, task_dict, processed_episodes, total_episodes, num_success, num_total)

    # Also write a concise summary file
    summary_path = args.save_dir / "evaluation_summary.json"
    summary_data = {
        'total_predictions': num_total,
        'num_success': num_success,
        'overall_success_rate': (num_success / num_total) if num_total > 0 else 0.0,
        'total_episodes': total_episodes,
        'processed_episodes': len(processed_episodes),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"Evaluation summary saved: {summary_path}")