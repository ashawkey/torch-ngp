#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_dnerf.py data/dnerf/bouncingballs --workspace trial_dnerf_bouncingballs -O --bound 1 --scale 0.8 --dt_gamma 0 #--gui --test
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_dnerf.py data/dnerf/bouncingballs --workspace trial_dnerf_basis_bouncingballs -O --bound 1 --scale 0.8 --dt_gamma 0 --basis #--gui --test

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_dnerf.py data/dnerf/standup --workspace trial_dnerf_standup -O --bound 1 --scale 0.8 --dt_gamma 0 #--gui --test

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_dnerf.py data/split-cookie/ --workspace trial_dnerf_cookies -O --bound 1 --scale 0.3 #--gui --test
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_dnerf.py data/split-cookie/ --workspace trial_dnerf_cookies_ncr --preload --fp16 --bound 1 --scale 0.3 #--gui --test

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_dnerf.py data/vrig-3dprinter/ --workspace trial_dnerf_printer -O --bound 2 --scale 0.33 #--gui --test