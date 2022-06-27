#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego -O --bound 1.0 --scale 0.67 --dt_gamma 0 --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego -O --bound 1.0 --scale 0.67 --dt_gamma 0 --gui --rand_pose 0 --clip_text "red" --lr 1e-3 --ckpt latest_model

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_llff_data/orchids --workspace trial_nerf_orchids -O --gui --bound 2.0 --scale 0.6
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_llff_data/orchids --workspace trial_nerf_orchids -O --gui --bound 2.0 --scale 0.6 --rand_pose 0 --clip_text "blue flower" --lr 1e-3 --ckpt latest_model