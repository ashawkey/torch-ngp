#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_fox -O --gui #--error_map
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego -O --bound 1.0 --scale 0.8 --dt_gamma 0 --gui #--error_map
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_llff_data/orchids --workspace trial_nerf_orchids -O --gui --bound 2.0 --scale 0.6
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/TanksAndTemple/Family --workspace trial_nerf_family -O --bound 1.0 --scale 0.33 --dt_gamma 0 --gui

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/figure --workspace trial_nerf_fig -O --gui --bound 1.0 --scale 0.3 --bg_radius 128
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python main_nerf.py data/vasedeck --workspace trial_nerf_vase -O --gui --bound 4.0 --scale 0.3