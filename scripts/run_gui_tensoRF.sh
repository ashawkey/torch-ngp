#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF --fp16 --cuda_ray --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego --fp16 --cuda_ray --bound 1.0 --scale 0.8 --mode blender --gui

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF_cp --cp --fp16 --cuda_ray --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego_cp --cp --fp16 --cuda_ray --bound 1.0 --scale 0.8 --mode blender --gui