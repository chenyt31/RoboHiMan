#!/bin/bash

# 默认参数值
DEFAULT_MAIN_DIR="path_to_save_file" # modify, e.g. HiMan_Data/train_logs_baseline_3dda
DEFAULT_DATASET="path_to_save_file" # modify, e.g. HiMan_Data/package_3dda_data
DEFAULT_VALSET="path_to_save_file" # modify, e.g. HiMan_Data/package_3dda_data
DEFAULT_CUDA="0"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --main_dir)
            main_dir="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --valset)
            valset="$2"
            shift 2
            ;;
        --cuda)
            cuda_visible="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置参数（如果未提供则使用默认值）
main_dir="${main_dir:-$DEFAULT_MAIN_DIR}"
dataset="${dataset:-$DEFAULT_DATASET}"
valset="${valset:-$DEFAULT_VALSET}"
cuda_visible="${cuda_visible:-$DEFAULT_CUDA}"

# 其他固定参数
lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=8
C=120
ngpus=1
quaternion_format=xyzw

# 执行训练命令
CUDA_VISIBLE_DEVICES=$cuda_visible CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/peract/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --num_workers 1 \
    --train_iters 600001 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 10000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 600 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr \
    --num_history $num_history \
    --cameras left_shoulder right_shoulder wrist front \
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --run_log_dir diffusion_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps
