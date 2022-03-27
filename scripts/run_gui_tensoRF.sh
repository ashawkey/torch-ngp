#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF --cuda_ray --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego --cuda_ray --bound 1.5 --scale 1.0 --mode blender --gui

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF_ff2 --fp16 --ff --cuda_ray --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego_ff2 --fp16 --ff --cuda_ray --bound 1.5 --scale 1.0 --mode blender --gui