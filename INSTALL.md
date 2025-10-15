# RoboHiMan

# Pre-requirement
```bash
apt install libxcb1 libxcb-shm0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxcb-render0 \
  libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0 libxkbcommon0

export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

```bash
cd path_to_RoboHiMan # modify

# we use uv to manage project dependencies and environments: https://docs.astral.sh/uv/
pip install uv # you can use pip to install uv or https://docs.astral.sh/uv/getting-started/installation/

# create env
uv venv --python 3.11

# run before pip install
source .venv/bin/activate
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# install PyRep, RLBench
cd baselines/RVT-Baseline
uv pip install wheel
uv pip install setuptools==69.0.3
uv pip install -r rvt/libs/PyRep/requirements.txt
uv pip install -e rvt/libs/PyRep --no-build-isolation
uv pip install -r rvt/libs/RLBench/requirements.txt
uv pip install -e rvt/libs/RLBench

# install HiMan-Bench version of robot-colosseum
cd ../..
cd HiMan-Bench/robot-colosseum
uv pip install -r requirements.txt
uv pip install -e .

uv pip install typed-argument-parser scipy opencv-python-headless yacs packaging pandas transformers blobfile ipykernel blosc
```


# Collect Dataset
```bash
cd path_to_RoboHiMan # modify

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/baselines/RVT-Baseline/rvt/libs/PyRep:$PWD/baselines/RVT-Baseline/rvt/libs/RLBench:$PWD/HiMan-Bench/robot-colosseum

# multi-level dataset
# to .sh file -> modify data save path
cd HiMan-Bench/robot-colosseum/
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash collect_dataset_train_A.sh
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash collect_dataset_train_AP.sh
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash collect_dataset_train_C.sh
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash collect_dataset_train_CP.sh

# rearrange the multi-level dataset if using L1
uv run src/data_preprocessing/multi_level_data_rearrange.py \
    --level1_dir HiMan_data/train_atomic_A \
    --save_dir HiMan_data/L1/train

# rearrange the multi-level dataset if using L4
uv run src/data_preprocessing/multi_level_data_rearrange.py \
    --level1_dir HiMan_data/train_atomic_A \
    --level2_dir HiMan_data/train_atomic_AP \
    --level3_dir HiMan_data/train_compositional_C \
    --level4_dir HiMan_data/train_compositional_CP \
    --save_dir HiMan_data/L4/train

# test
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash collect_dataset_test_atomic.sh
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" bash collect_dataset_test_compositional.sh
```

# Train Model
- [RVT-Baseline](baselines/RVT-Baseline/README.md)
- [3D-Diffuser-Actor-Baseline](baselines/3d-Diffuser-Actor-Baseline/README.md)
- [OpenPi-Baseline](baselines/OpenPi-Baseline/README.md)
- [OpenPi05-Baseline](baselines/OpenPi05-Baseline/README.md)

# Acknowledgment
This project is based on the following repositories:
- [RVT](https://github.com/nvlabs/rvt)
- [3D-Diffuser-Actor](https://github.com/nickgkan/3d_diffuser_actor)
- [OpenPi](https://github.com/Physical-Intelligence/openpi)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Colosseum](https://github.com/robot-colosseum/robot-colosseum)
- [DeCo](https://deco226.github.io/)