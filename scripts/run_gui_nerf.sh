#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_ff2 -O --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_ff2 -O --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --gui

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/figure --workspace trial_nerf_figure -O --gui --scale 0.33 --bound 1.0