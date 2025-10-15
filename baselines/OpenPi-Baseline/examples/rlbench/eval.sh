#!/bin/bash

# === Usage Function ===
usage() {
  echo "Usage: bash eval_with_mode.sh --host <host> --port <port> --max_steps <steps> \
  --num_episodes <episodes> --tasks_type <atomic|compositional> --output_file <path> \
  --data_dir <path> --eval_mode <vanilla|half|vlm>"
  exit 1
}

# === Argument Parsing ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --host) host="$2"; shift ;;
        --port) port="$2"; shift ;;
        --max_steps) max_steps="$2"; shift ;;
        --num_episodes) num_episodes="$2"; shift ;;
        --tasks_type) tasks_type="$2"; shift ;;
        --output_file) output_file="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --eval_mode) eval_mode="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# === Check Required Params ===
if [[ -z "$host" || -z "$port" || -z "$max_steps" || -z "$num_episodes" || 
      -z "$tasks_type" || -z "$output_file" || -z "$data_dir" || -z "$eval_mode" ]]; then
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
echo "Starting evaluation for task type: $tasks_type"
echo "Using data directory: $data_dir"

for root_task in "${root_tasks[@]}"; do
    for i in {0..17}; do
        task_name="${root_task}_${i}"
        DATA_PATH="${data_dir}/${task_name}/"

        if [ ! -d "$DATA_PATH" ]; then
            echo "[Skip] $DATA_PATH does not exist."
            continue
        fi

        python examples/rlbench/main.py \
            --args.tasks="$task_name" \
            --args.host="$host" \
            --args.port="$port" \
            --args.max_steps="$max_steps" \
            --args.data_dir="$data_dir" \
            --args.num_episodes="$num_episodes" \
            --args.tasks_type="$tasks_type" \
            --args.output_file="$output_file" \
            --args.eval_mode="$eval_mode"
    done
done

echo "Evaluation completed for task type: $tasks_type"
