#!/bin/bash

# === Usage Function ===
usage() {
  echo "Usage: bash run_eval.sh --output_file <path> --data_dir <path> \
  --tasks_type <atomic|compositional> --eval_mode <vanilla|half|vlm> \
  --checkpoint <path> --num_episodes <episodes> --max_steps <steps>"
  exit 1
}

# === Argument Parsing ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_file) output_file="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --tasks_type) tasks_type="$2"; shift ;;
        --eval_mode) eval_mode="$2"; shift ;;
        --checkpoint) checkpoint="$2"; shift ;;
        --num_episodes) num_episodes="$2"; shift ;;
        --max_steps) max_steps="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# === Check Required Params ===
if [[ -z "$output_file" || -z "$data_dir" || -z "$tasks_type" || -z "$eval_mode" || -z "$checkpoint" || -z "$num_episodes" || -z "$max_steps" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ "$tasks_type" != "atomic" && "$tasks_type" != "compositional" ]]; then
    echo "Error: Invalid tasks_type. Must be 'atomic' or 'compositional'."
    usage
fi

if [[ "$eval_mode" != "vanilla" && "$eval_mode" != "half" && "$eval_mode" != "vlm" ]]; then
    echo "Error: Invalid eval_mode. Must be 'vanilla', 'half', or 'vlm'."
    usage
fi

# === Fixed Parameters (Keep Original) ===
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="left_shoulder,right_shoulder,wrist,front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=0
quaternion_format=xyzw
exp="3d_diffuser_actor_${tasks_type}"

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
else  # compositional
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

        CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy_himan.py \
        --tasks "$task_name" \
        --checkpoint "$checkpoint" \
        --diffusion_timesteps 100 \
        --fps_subsampling_factor $fps_subsampling_factor \
        --lang_enhanced $lang_enhanced \
        --relative_action $relative_action \
        --num_history 3 \
        --test_model 3d_diffuser_actor \
        --cameras $cameras \
        --verbose $verbose \
        --action_dim 8 \
        --collision_checking 0 \
        --predict_trajectory 1 \
        --embedding_dim $embedding_dim \
        --rotation_parametrization "6D" \
        --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
        --data_dir "$data_dir" \
        --num_episodes "$num_episodes" \
        --output_file "$output_file" \
        --use_instruction $use_instruction \
        --instructions instructions/peract/instructions.pkl \
        --variations {0..60} \
        --max_tries $max_tries \
        --max_steps "$max_steps" \
        --seed $seed \
        --gripper_loc_bounds_file $gripper_loc_bounds_file \
        --gripper_loc_bounds_buffer 0.04 \
        --quaternion_format $quaternion_format \
        --interpolation_length $interpolation_length \
        --dense_interpolation 1 \
        --headless 1 \
        --eval_mode "$eval_mode" \
        --tasks_type "$tasks_type"
    done
done