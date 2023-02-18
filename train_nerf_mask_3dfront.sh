#!/usr/bin/env bash

set -x
set -e

python3 main_nerf_mask.py \
/data2/jhuangce/3dfront_mask_data/masks_2d/3dfront_0019_00 \
--workspace ./workspace/3dfront_nerf_mask/3dfront_0019_00 \
--iters 10000 \
--lr 1e-2 \
--ckpt ./workspace/3dfront_nerf/3dfront_0019_00/checkpoints/ngp_ep0055.pth \
--load_model_only \
--train_mask
