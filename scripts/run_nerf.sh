#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf_ff --fp16 --ff
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf_tcnn --fp16 --tcnn --cuda_raymarching
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego_tcnn --bound 1 --fp16 --tcnn
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego_ff --bound 1 --fp16 --ff