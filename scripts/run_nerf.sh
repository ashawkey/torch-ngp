#! /bin/bash

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf --fp16 --preload
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_ff --fp16 --ff --preload
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_tcnn --fp16 --tcnn --preload

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf2 --fp16 --cuda_ray --preload
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_ff2 --fp16 --ff --cuda_ray --preload
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_tcnn2 --fp16 --tcnn --cuda_ray --preload

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego --fp16 --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_ff --fp16 --ff --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_tcnn --fp16 --tcnn --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego2 --fp16 --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_ff2 --fp16 --ff --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload --error_map
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_tcnn2 --fp16 --tcnn --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload