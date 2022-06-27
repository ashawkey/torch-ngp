#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_dnerf.py data/dnerf/bouncingballs --workspace trial_dnerf_bouncingballs -O --bound 1 --scale 0.8 --dt_gamma 0 #--gui --test
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_dnerf.py data/dnerf/jumpingjacks --workspace trial_dnerf_jumpingjacks -O --bound 1 --scale 0.8 --dt_gamma 0 #--gui --test

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_dnerf.py data/slice-banana --workspace trial_dnerf_slicebanana -O --bound 1 --scale 1 --dt_gamma 0 #--gui --test