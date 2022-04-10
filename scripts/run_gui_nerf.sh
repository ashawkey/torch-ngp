#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf --cuda_ray --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --gui

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_ff2 --fp16 --ff --cuda_ray --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_ff2 --fp16 --ff --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --gui

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/figure --workspace trial_nerf_figure --fp16 --cuda_ray --gui --scale 0.33 --bound 1.0