#!/bin/bash

# === Usage Function ===
usage() {
  echo "Usage: bash eval.sh --epoch <epoch> --model_folder <path> --device <device> \
  --tasks_type <atomic|compositional> --eval_mode <vanilla|half|vlm> \
  --data_dir <path> --num_episodes <episodes> --log_name <name> \
  [--high_level_cfg_path <path>]"
  exit 1
}

# === Argument Parsing ===
# 初始化变量
high_level_cfg_path=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epoch) epoch="$2"; shift ;;
        --model_folder) model_folder="$2"; shift ;;
        --device) device="$2"; shift ;;
        --tasks_type) tasks_type="$2"; shift ;;
        --eval_mode) eval_mode="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --num_episodes) num_episodes="$2"; shift ;;
        --log_name) log_name="$2"; shift ;;
        --high_level_cfg_path) high_level_cfg_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# === Check Required Params ===
if [[ -z "$epoch" || -z "$model_folder" || -z "$device" || -z "$tasks_type" || -z "$eval_mode" || -z "$data_dir" || -z "$num_episodes" || -z "$log_name" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# === Validate Params ===
if [[ "$tasks_type" != "atomic" && "$tasks_type" != "compositional" ]]; then
    echo "Error: Invalid tasks_type. Must be 'atomic' or 'compositional'."
    usage
fi

if [[ "$eval_mode" != "vanilla" && "$eval_mode" != "half" && "$eval_mode" != "vlm" ]]; then
    echo "Error: Invalid eval_mode. Must be 'vanilla', 'half', or 'vlm'."
    usage
fi

# === Select Tasks ===
if [[ "$tasks_type" == "atomic" ]]; then
    root_tasks=(
        "box_in_cupboard"
        "box_out_of_opened_drawer"
        "close_drawer"
        "put_in_opened_drawer"
        "sweep_to_dustpan"
        "box_out_of_cupboard"
        "broom_out_of_cupboard"
        "open_drawer"
        "rubbish_in_dustpan"
        "take_out_of_opened_drawer"
    )
else
    root_tasks=(
        "box_exchange"
        "put_in_and_close"
        "put_in_without_close"
        "put_two_in_different"
        "put_two_in_same"
        "retrieve_and_sweep"
        "sweep_and_drop"
        "take_out_and_close"
        "take_out_without_close"
        "take_two_out_of_different"
        "take_two_out_of_same"
        "transfer_box"
    )
fi

# === Evaluation Loop ===
for root_task in "${root_tasks[@]}"; do
    for i in {0..17}; do
        task_name="${root_task}_${i}"
        DATA_PATH="${data_dir}/${task_name}/"
        
        if [ ! -d "$DATA_PATH" ]; then
            echo "[Skip] $DATA_PATH does not exist."
            continue
        fi

        cmd_args=(
            uv run eval.py
            --model-folder "$model_folder"
            --eval-datafolder "$data_dir"
            --tasks "$task_name"
            --eval-episodes "$num_episodes"
            --log-name "$log_name"
            --device "$device"
            --headless
            --model-name "model_${epoch}.pth"
            --save-video
            --tasks_type "$tasks_type"
            --eval_mode "$eval_mode"
        )

        # 仅当提供了high_level_cfg_path时才添加该参数
        if [ -n "$high_level_cfg_path" ]; then
            cmd_args+=(--high_level_cfg_path "$high_level_cfg_path")
        fi

        echo "[Run] Evaluating $task_name ..."
        "${cmd_args[@]}"
    done
done
