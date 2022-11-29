#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_sdf.py data/armadillo.obj --workspace trial_sdf_ff --fp16 --ff
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_sdf.py data/armadillo.obj --workspace trial_sdf_tcnn --fp16 --tcnn

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_sdf.py data/lucy.obj --workspace trial_sdf --fp16