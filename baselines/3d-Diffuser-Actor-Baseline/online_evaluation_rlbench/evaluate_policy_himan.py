"""Online evaluation script on RLBench."""
import datetime
import random
from typing import Tuple, Optional
from pathlib import Path
import json
import os

from omegaconf import OmegaConf
import torch
import numpy as np
import tap

from diffuser_actor.keypose_optimization.act3d import Act3D
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from src.high_level.vlm.agent import MultiModalAgent
from src.high_level.vlm.models.qwen_vl_runner import QwenVLRunner
from utils.common_utils import (
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)
from utils.utils_with_rlbench import RLBenchEnv, Actioner, load_episodes


class Arguments(tap.Tap):
    checkpoint: Path = ""
    seed: int = 2
    device: str = "cuda"
    start_episode: int = 0
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    tasks: Tuple[str, ...] = ("close_jar",)
    instructions: Optional[Path] = "instructions.pkl"
    variations: Tuple[int, ...] = (-1,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    max_steps: int = 25
    test_model: str = "3d_diffuser_actor"
    collision_checking: int = 0
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    gripper_loc_bounds_buffer: float = 0.04
    single_task_gripper_loc_bounds: int = 0
    predict_trajectory: int = 1

    # Act3D model parameters
    num_query_cross_attn_layers: int = 2
    num_ghost_point_cross_attn_layers: int = 2
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    regress_position_offset: int = 0

    # 3D Diffuser Actor model parameters
    diffusion_timesteps: int = 100
    num_history: int = 3
    fps_subsampling_factor: int = 5
    lang_enhanced: int = 0
    dense_interpolation: int = 1
    interpolation_length: int = 2
    relative_action: int = 0

    # Shared model parameters
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 1
    rotation_parametrization: str = '6D'
    quaternion_format: str = 'xyzw'

    # himan
    tasks_type: str = "atomic"
    eval_mode: str = "vanilla"
    high_level_cfg_path: str = ""

def save_video(vid, save_path, fps=10):
    import imageio
    vid = vid.transpose(0, 2, 3, 1)  # (T, H, W, C)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, vid, fps=fps)

def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print('Gripper workspace')
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_loc_bounds_buffer,
    )

    if args.test_model == "3d_diffuser_actor":
        model = DiffuserActor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
        ).to(device)
    elif args.test_model == "act3d":
        model = Act3D(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=(
                args.num_ghost_point_cross_attn_layers),
            num_query_cross_attn_layers=(
                args.num_query_cross_attn_layers),
            num_vis_ins_attn_layers=(
                args.num_vis_ins_attn_layers),
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=(
                args.fine_sampling_ball_diameter),
            regress_position_offset=bool(
                args.regress_position_offset),
            use_instruction=bool(args.use_instruction)
        ).to(device)
    else:
        raise NotImplementedError

    # Load model weights
    model_dict = torch.load(args.checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)
    model.eval()

    return model


if __name__ == "__main__":
    # Arguments
    args = Arguments().parse_args()
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    # print("Arguments:")
    # print(args)
    # print("-" * 100)
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    log_dirpath = os.path.dirname(args.output_file)
    txt_log_path = os.path.join(log_dirpath, "episode_log.txt")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)

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
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking),
        tasks_type=args.tasks_type,
        tasks=args.tasks,
        vlm_agent=vlm_agent if args.eval_mode == "vlm" else None,
        tmp_dir = high_level_cfg.tmp_dir if args.eval_mode == "vlm" else ""
    )

    actioner = Actioner(
        policy=model,
        instructions=None,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory)
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    env.env.launch(args.tasks_type)

    num_tasks = len(args.tasks)
    for task_id in range(num_tasks):
        task_name = args.tasks[task_id]
        success_list = []
        for ep in range(args.start_episode, args.start_episode + args.num_episodes):
            sr, vid = env._evaluate_task_on_one_variation(
                task_str=task_name,
                eval_demo_seed=ep,
                max_steps=args.max_steps,
                actioner=actioner,
                max_tries=args.max_tries,
                verbose=args.verbose,
                interpolation_length=args.interpolation_length,
                num_history=args.num_history,
                eval_mode=args.eval_mode
            )
            # log eps info
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp} | [Task {task_name}] Episode {ep} -> Success={sr}\n"
            print(log_line.strip())
            with open(txt_log_path, "a") as f:
                f.write(log_line)

            # log video
            video_dir = os.path.join(log_dirpath, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{task_name}_ep{ep}_{sr}.mp4")
            save_video(vid, video_path, fps=15)
            print(f'save video in {video_path}')

            success_list.append(sr)
    
        # log avg. sr to output file
        avg_sr = float(np.mean(success_list))
        avg_line = f"[{task_name}]: {avg_sr:.3f}\n"
        print(avg_line.strip())
        with open(args.output_file, "a") as f:
            f.write(avg_line)

    env.env.shutdown()
    

    