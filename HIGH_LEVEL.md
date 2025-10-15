# 🧠 High-Level Policy (LLaMA-Factory) Setup & Usage

This guide covers **environment setup**, **dataset preparation**, **training**, and **evaluation** for the high-level VLM-based policy in RoboHiMan.


## 1️⃣ Install High-Level Policy

```bash
# Load environment
source ~/.bashrc

# Go to LLaMA-Factory
cd third_party/LLaMA-Factory

# Activate virtual environment
# uv venv .llama-factory
source .llama-factory/bin/activate

# Set CUDA path
export CUDA_HOME=/usr/local/cuda-11.8

# Install PyTorch & dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
uv pip install -e ".[torch,metrics]" --no-build-isolation

# Install additional Python packages
uv pip install tqdm tap jinja2
uv pip install pathlib typing pickle json collections
```


## 2️⃣ Dataset Preparation

L1 Dataset
```bash
uv run src/high_level/data_preprocessing/planning_sft_data.py \
    --data_dir <PATH_TO_L1_TRAIN_DATA> \
    --save_dir <PATH_TO_SAVE_PROCESSED_DATA> \
    --gen_cot_vlm_ckpt_dir <PATH_TO_VLM_CKPT> \
    --cot_vlm_device cuda:0 \
    --prompt_dir src/high_level/prompts
```
Replace <PATH_TO_L1_TRAIN_DATA>, <PATH_TO_SAVE_PROCESSED_DATA>, and <PATH_TO_VLM_CKPT> with your paths.


## 3️⃣ Training

### 3.1 Environment Setup
```bash
source ~/.bashrc
cd <PATH_TO_LLaMA_FACTORY>
# uv venv .llama-factory
source .llama-factory/bin/activate
```

### 3.2 Train Full Model
```bash
CUDA_VISIBLE_DEVICES=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml \
    model_name_or_path=<PATH_TO_BASE_CKPT> \
    dataset=Less_L4_Base_History \
    output_dir=<PATH_TO_OUTPUT_DIR> \
    deepspeed=examples/deepspeed/ds_z3_config.json \
    per_device_train_batch_size=8 \
    gradient_accumulation_steps=8 \
    num_train_epochs=5
```

### 3.3 Train LoRA Adapter
```bash
CUDA_VISIBLE_DEVICES=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_v1_L1-L4.yaml \
    model_name_or_path=<PATH_TO_BASE_CKPT> \
    dataset=Less_L4_Base_History \
    output_dir=<PATH_TO_OUTPUT_DIR> \
    per_device_train_batch_size=8 \
    gradient_accumulation_steps=8 \
    num_train_epochs=5
```

Merge LoRA Adapter (Optional)
```bash
llamafactory-cli export examples/merge_lora/qwen2_5vl_lora_sft.yaml \
    model_name_or_path=<PATH_TO_BASE_CKPT> \
    adapter_name_or_path=<PATH_TO_LORA_CKPT_DIR> \
    export_dir=<PATH_TO_MERGED_OUTPUT_DIR>
```


## 4️⃣ Evaluation

### 4.1 Environment Setup
```bash
cd <PATH_TO_ROBOHIMAN_ROOT>
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/third_party/RVT/rvt/libs/PyRep:$PWD/third_party/RVT/rvt/libs/RLBench:$PWD/third_party/RVT/rvt/libs/point-renderer:$PWD/third_party/RVT/rvt/libs/peract_colab:$PWD/third_party/RVT/rvt/libs/YARR:$PWD/third_party/robot-colosseum:$PWD/third_party/RVT/rvt/libs/peract:$PWD/third_party/RVT/rvt/libs/peract_colab
cd third_party/LLaMA-Factory/
source .llama-factory/bin/activate
cd ../..
export COPPELIASIM_ROOT=<PATH_TO_COPPELIASIM>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

## 4.2 Evaluate 7B Model

Atomic Tasks
```bash
uv run src/high_level/eval.py \
    --data_dir <PATH_TO_TEST_ATOMIC> \
    --save_dir <PATH_TO_EVAL_OUTPUT_ATOMIC> \
    --vlm_ckpt_dir <PATH_TO_VLM_CKPT> \
    --vlm_device cuda:1 \
    --prompt_dir src/high_level/prompts
```

Compositional Tasks
```bash
uv run src/high_level/eval.py \
    --data_dir <PATH_TO_TEST_COMPOSITIONAL> \
    --save_dir <PATH_TO_EVAL_OUTPUT_COMPOSITIONAL> \
    --vlm_ckpt_dir <PATH_TO_VLM_CKPT> \
    --vlm_device cuda:1 \
    --frame_interval 30 \
    --prompt_dir src/high_level/prompts
```


## 5️⃣ Low-Level Policy with VLM-based Planner

```bash
# L1
cd third_party/LLaMA-Factory
source .llama-factory/bin/activate
# inference
API_PORT=9001 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/qwen2_5vl.yaml \
model_name_or_path=<PATH_TO_VLM_CKPT> 
```


Refer to [RVT-Baseline](baselines/RVT-Baseline/README.md) for low-level policy env installation
```bash
cd baselines/RVT-Baseline
source .rvt-baseline/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/libs/PyRep:$PWD/libs/RLBench:$PWD/libs/point-renderer:$PWD/libs/peract_colab:$PWD/libs/YARR:$PWD/libs/peract:$PWD/../../HiMan-Bench/robot-colosseum
cd rvt
```

```bash
# L1 Atomic
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash eval.sh \
    --epoch last \
    --model_folder <PATH_TO_RVT_L1_CKPT> \
    --device 0 \
    --tasks_type atomic \
    --eval_mode vlm \
    --high_level_cfg_path <PATH_TO_HIGH_LEVEL_CFG_L1> \
    --data_dir <PATH_TO_TEST_ATOMIC> \
    --num_episodes 5 \
    --log_name "Base_VLM_Atomic"
```

```bash
# L1 compositional
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash eval.sh \
    --epoch last \
    --model_folder <PATH_TO_RVT_L1_CKPT> \
    --device 0 \
    --tasks_type compositional \
    --eval_mode vlm \
    --high_level_cfg_path <PATH_TO_HIGH_LEVEL_CFG_L1> \
    --data_dir <PATH_TO_TEST_COMPOSITIONAL> \
    --num_episodes 5 \
    --log_name "Base_VLM_Compositional"
```