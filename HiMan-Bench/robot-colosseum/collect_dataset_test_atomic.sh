#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    echo "Collecting demos from all tasks"
    tasks=("box_in_cupboard"
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
    echo "Collectins demos from task $1"
    tasks=("$1")
fi

# idx from which to collect demos (use -1 for all idxs)
IDX_TO_COLLECT=-1

SAVE_PATH=HiMan_data/test_atomic # modify
NUMBER_OF_EPISODES=1 # modify
IMAGE_SIZE=(128 128)
MAX_ATTEMPTS=20
SEED=244
USE_SAVE_STATES="True"

IMAGES_USE_RGB="True"
IMAGES_USE_DEPTH="True"
IMAGES_USE_MASK="False"
IMAGES_USE_POINTCLOUD="False"

CAMERAS_USE_LEFT_SHOULDER="True"
CAMERAS_USE_RIGHT_SHOULDER="True"
CAMERAS_USE_OVERHEAD="False"
CAMERAS_USE_WRIST="True"
CAMERAS_USE_FRONT="True"

for task in "${tasks[@]}"
do
    echo "Processing task: $task"
    python -m colosseum.tools.dataset_generator_atomic --config-name $task \
            env.seed=$SEED \
            data.save_path=$SAVE_PATH \
            +data.max_attempts=$MAX_ATTEMPTS \
            +data.idx_to_collect=$IDX_TO_COLLECT \
            +data.use_save_states=$USE_SAVE_STATES \
            data.image_size=[${IMAGE_SIZE[0]},${IMAGE_SIZE[1]}] \
            data.episodes_per_task=$NUMBER_OF_EPISODES \
            data.images.rgb=$IMAGES_USE_RGB \
            data.images.depth=$IMAGES_USE_DEPTH \
            data.images.mask=$IMAGES_USE_MASK \
            data.images.point_cloud=$IMAGES_USE_POINTCLOUD \
            data.cameras.left_shoulder=$CAMERAS_USE_LEFT_SHOULDER \
            data.cameras.right_shoulder=$CAMERAS_USE_RIGHT_SHOULDER \
            data.cameras.overhead=$CAMERAS_USE_OVERHEAD \
            data.cameras.wrist=$CAMERAS_USE_WRIST \
            data.cameras.front=$CAMERAS_USE_FRONT
done
