from dataclasses import dataclass
import datetime
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

from examples.rlbench.rlbench_env import RLBenchEnv
from examples.rlbench.rlbench_utils import Actioner

from src.high_level.vlm.agent import MultiModalAgent
from src.high_level.vlm.models.qwen_vl_runner import QwenVLRunner

@dataclass
class Args:
    # Model parameters
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 25

    # Environment parameters
    data_dir: str | None = ""  # Path to the dataset
    random_seed: int = 42  # Random seed for the environment
    tasks: tuple[str, ...] = ("close_jar",)  # Tasks to evaluate, comma separated
    apply_cameras: str = "left_shoulder,right_shoulder,wrist,front"  # Cameras to use, comma separated
    start_episode: int = 0
    num_episodes: int = 1  # Number of evaluation episodes per task
    max_steps: int = 300  # Maximum steps per episode
    headless: bool = True  # Run headless
    image_size: str = "256,256"  # Image size (width,height)
    tasks_type: str = "atomic"

    # Utility settings
    verbose: bool = True  # Verbose output
    output_file: Path = Path(__file__).parent / "eval.json"  # log file
    eval_mode: str = "vanilla"

    high_level_cfg_path: str = ""

def save_video(vid, save_path: Path, fps=10):
    import imageio

    vid = vid.transpose(0, 2, 3, 1)  # (T, H, W, C)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, vid, fps=fps)


def eval_rlbench(args: Args) -> None:
    # Resolve output file path and create parent directory
    args.output_file = args.output_file.resolve()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    log_dirpath = args.output_file.parent
    txt_log_path = log_dirpath / "episode_log.txt"

    # Validate and resolve data_dir if provided
    if args.data_dir and args.data_dir.strip():
        args.data_dir = Path(args.data_dir).resolve()
        if not args.data_dir.is_dir():
            raise NotADirectoryError(f"Specified data_dir is not a valid directory: {args.data_dir}")

    # Set random seeds
    seed = args.random_seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse image size
    try:
        image_size = [int(x.strip()) for x in args.image_size.split(",")]
        if len(image_size) != 2:
            raise ValueError("image_size must contain exactly two values (width, height)")
    except ValueError as e:
        raise ValueError(f"Invalid image_size format: {args.image_size}. Use 'width,height' (e.g., '256,256')") from e

    # Parse apply_cameras
    apply_cameras = [cam.strip() for cam in args.apply_cameras.split(",") if cam.strip()]
    if not apply_cameras:
        raise ValueError("apply_cameras cannot be empty. Provide at least one camera (e.g., 'front,wrist')")

    # evaluate agent
    if args.eval_mode == "vlm":
        if args.high_level_cfg_path is None: # high_level_cfg_path is required
            raise ValueError("high_level_cfg_path is required")
        high_level_cfg = OmegaConf.load(args.high_level_cfg_path)
        vlm_runner = QwenVLRunner(
            mode=high_level_cfg.mode, 
            model_name=high_level_cfg.vlm_ckpt_dir, 
            device=high_level_cfg.vlm_device,
            ip=high_level_cfg.ip,
            port=high_level_cfg.port,
        )
        vlm_agent = MultiModalAgent(
            runner=vlm_runner, 
            base_dir=high_level_cfg.prompt_dir, 
            prompt_model_name=high_level_cfg.prompt_model_name
        )

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=str(args.data_dir) if args.data_dir else None,
        image_size=image_size,
        apply_rgb=True,
        apply_pc=True,
        headless=args.headless,
        apply_cameras=apply_cameras,
        tasks_type=args.tasks_type,
        tasks=list(args.tasks),
        vlm_agent=vlm_agent if args.eval_mode == "vlm" else None,
        tmp_dir = high_level_cfg.tmp_dir if args.eval_mode == "vlm" else ""
    )

    # Create Actioner
    actioner = Actioner(
        host=args.host,
        port=args.port,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
    )

    env.env.launch(args.tasks_type)
    num_tasks = len(args.tasks)
    for task_id in range(num_tasks):
        task_name = args.tasks[task_id]
        success_list = []
        for ep in range(args.start_episode, args.start_episode + args.num_episodes):
            sr, vid = env.evaluate_task_on_one_variation(
                task_str=task_name,
                eval_demo_seed=ep,
                max_steps=args.max_steps,
                actioner=actioner,
                verbose=args.verbose,
                eval_mode=args.eval_mode,
            )
            # Log episode info
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp} | [Task {task_name}] Episode {ep} -> Success={sr}\n"
            print(log_line.strip())
            with txt_log_path.open("a") as f:
                f.write(log_line)

            # Save video
            video_dir = log_dirpath / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{task_name}_ep{ep}_{sr}.mp4"
            save_video(vid, video_path, fps=15)
            print(f"Saved video to {video_path}")

            success_list.append(sr)

        # Log average success rate to output file
        avg_sr = float(np.mean(success_list)) if success_list else 0.0
        avg_line = f"[{task_name}]: {avg_sr:.3f}\n"
        print(avg_line.strip())
        with args.output_file.open("a") as f:
            f.write(avg_line)

    env.env.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_rlbench)
