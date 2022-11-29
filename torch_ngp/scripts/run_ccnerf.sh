#! /bin/bash

# train single objects
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python main_CCNeRF.py data/nerf_synthetic/ficus --workspace trial_cc_ficus -O --bound 1.0 --scale 0.67 --dt_gamma 0 --error_map
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python main_CCNeRF.py data/nerf_synthetic/chair --workspace trial_cc_chair -O --bound 1.0 --scale 0.67 --dt_gamma 0 --error_map
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python main_CCNeRF.py data/nerf_synthetic/hotdog --workspace trial_cc_hotdog -O --bound 1.0 --scale 0.67 --dt_gamma 0 --error_map

# compose
# use more samples per ray (--max_steps) and a larger bound for better results
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python main_CCNeRF.py data/nerf_synthetic/hotdog --workspace trial_cc_hotdog -O --bound 2.0 --scale 0.67 --dt_gamma 0 --max_steps 2048 --test --compose