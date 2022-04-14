#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/fox --workspace trial_nerf_fox -O --gui --error_map
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf_lego -O --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --gui #--error_map
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/nerf_llff_data/orchids --workspace trial_nerf_orchids -O --gui --bound 2.0 --scale 0.6

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py data/figure --workspace trial_nerf_fig -O --gui --bound 1.0 --scale 0.3

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/garden --workspace trial_nerf_garden --fp16 --cuda_ray --gui --bound 8.0 --scale 1.0