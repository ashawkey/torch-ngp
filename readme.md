# HashGrid Encoder (WIP)

A pytorch implementation of the HashGrid Encoder from [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

# Progress

* HashGrid Encoder
    - [*] basic pytorch CUDA extension implementation
    - [] speed optimization
* Experiments
    - [*] SDF
    - [] NeRF

# Usage

```bash
# SDF experiment
CUDA_VISIBLE_DEVICES=1 python train_sdf.py data/armadillo.obj --workspace trial
```