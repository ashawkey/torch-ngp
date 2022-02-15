#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python train_sdf.py data/armadillo.obj --workspace trial_sdf_ff --fp16 --ff
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python train_sdf.py data/armadillo.obj --workspace trial_sdf_tcnn --fp16 --tcnn