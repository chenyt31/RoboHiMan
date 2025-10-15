uv run data_preprocessing/package_himan.py \
    --data_dir="path_to_data" \ # modify, e.g. HiMan_Data/train
    --output="path_to_save_file" \ # modify, e.g. HiMan_Data/package_3dda_data
    --device="cuda:0"