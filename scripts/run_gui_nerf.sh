#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python gui_nerf.py data/fox --workspace trial_nerf_fox_ff2 --fp16 --ff --cuda_ray --train
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python gui_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_ff2 --fp16 --ff --cuda_ray --bound 1 --scale 0.8 --mode blender --train