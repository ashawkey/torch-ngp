#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF_fox -O --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego -O --bound 1.0 --scale 0.8 --dt_gamma 0 --gui

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensorCP_fox --cp -O --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensorCP_lego --cp -O --bound 1.0 --scale 0.8 --dt_gamma 0 --gui

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/figure --workspace trial_tensoRF_fig -O --gui --scale 0.33 --bound 1.0 --bg_radius 32