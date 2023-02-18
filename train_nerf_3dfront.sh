#!/usr/bin/env bash

set -x
set -e

python3 main_nerf_mask.py \
/data2/jhuangce/torch-ngp/FRONT3D_render/finished/3dfront_0019_00/train \
--workspace ./workspace/3dfront_nerf/3dfront_0019_00 \
--iters 10000 \
--lr 1e-2 \
--cuda_ray
