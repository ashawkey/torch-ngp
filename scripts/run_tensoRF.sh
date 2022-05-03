#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF --fp16 --preload
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF2 --fp16 --cuda_ray --preload

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF_CP --fp16 --cp --preload
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/fox --workspace trial_tensoRF_CP2 --fp16 --cuda_ray --cp --preload

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego --num_steps 1024 --fp16 --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego2 --fp16 --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload #--error_map

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_CP_lego --cp --resolution1 500 --fp16 --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender  --preload
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_CP_lego2 --cp --resolution1 500 --fp16 --cuda_ray --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --preload

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/figure --workspace trial_tensoRF_fig -O --scale 0.33 --bound 1.0