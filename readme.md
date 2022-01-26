# HashGrid Encoder (WIP)

A pytorch implementation of the HashGrid Encoder from [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

**Note**: This repo only tries to implement the hash grid encoder for now, and is far from instant (especially for NeRF experiments).
The major time bottleneck now is the MLP implementation.

SDF | NeRF
:---: | :---:
![](assets/armadillo.jpg) | ![](assets/fox.gif)

# Progress

* HashGrid Encoder
    - [x] basic pytorch CUDA extension
    - [x] fp16 support 
* Experiments
    - SDF
        - [x] baseline
        - [ ] better SDF calculation (especially for non-watertight meshes)
    - NeRF
        - [x] baseline (although much slower)
        - [ ] ray marching in CUDA.

# News
* 1.26: add fp16 support for HashGrid Encoder (requires CUDA >= 10 and GPU ARCH >= 70 for now...).

# Install
```bash
pip install -r requirements.txt
```
Tested on Ubuntu with torch 1.10 & CUDA 11.3

# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

```bash
# SDF experiment
bash scripts/run_sdf.sh

# NeRF experiment
bash scripts/run_nerf.sh
```