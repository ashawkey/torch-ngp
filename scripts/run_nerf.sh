#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf --fp16
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf_ff --fp16 --ff
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf_tcnn --fp16 --tcnn

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf2 --fp16 --cuda_ray
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf_ff2 --fp16 --ff --cuda_ray
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/fox/transforms.json --workspace trial_nerf_tcnn2 --fp16 --tcnn --cuda_ray

# for the nerf synthetic dataset, you need to manually complete the json config, since I want (am) to (too) keep (lazy) the dataset code simple for now... ;)
# e.g. for lego, add the following to transforms_train.json:
# "fl_x": 1111,
# "fl_y": 1111,
# "h": 800,
# "w": 800,
# "cx": 400,
# "cy": 400,

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego --bound 1 --fp16
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego_ff --bound 1 --fp16 --ff 
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego_tcnn --bound 1 --fp16 --tcnn

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego2 --bound 1 --fp16 --cuda_ray
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego_ff2 --bound 1 --fp16 --ff --cuda_ray
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_nerf.py data/nerf_synthetic/lego/transforms_train.json --workspace trial_nerf_lego_tcnn2 --bound 1 --fp16 --tcnn --cuda_ray