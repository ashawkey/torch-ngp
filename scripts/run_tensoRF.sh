#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF --fp16
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF2 --fp16 --cuda_ray

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF_CP --fp16 --cp
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF_CP2 --fp16 --cuda_ray --cp

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego --fp16 --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego2 --fp16 --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_CP_lego --cp --resolution1 500 --fp16 --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender 
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_CP_lego2 --fp16 --cuda_ray --cp --resolution1 500 --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender