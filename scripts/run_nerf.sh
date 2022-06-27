#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/fox --workspace trial_nerf_fox -O
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego -O --bound 1 --scale 0.8 --dt_gamma 0
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego_emap -O --bound 1 --scale 0.8 --dt_gamma 0 --error_map
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/TanksAndTemple/Barn --workspace trial_nerf_barn -O --bound 1.0 --scale 0.33 --dt_gamma 0

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/firekeeper --workspace trial_nerf_firekeeper_bg_32 -O --bound 1.0 --scale 0.33 --bg_radius 32 #--gui #--test
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/garden --workspace trial_nerf_garden_bound_16 --cuda_ray --fp16 --bound 16.0 --scale 0.33 #--gui --test

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/vasedeck --workspace trial_nerf_vasedeck -O --bound 4.0 --scale 0.33 #--gui #--test
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/vasedeck --workspace trial_nerf_vasedeck_bg_32 -O --bound 4.0 --scale 0.33 --bg_radius 32 #--gui #--test