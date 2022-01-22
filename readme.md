# HashGrid Encoder (WIP)

A pytorch implementation of the HashGrid Encoder from [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

**Note**: This repo only tries to implement the hash grid encoder for now, and is far from instant (especially for NeRF experiments).

![](assets/fox.gif)
![](assets/armadillo.jpg)

# Progress

* HashGrid Encoder
    - [x] basic pytorch CUDA extension
    - [ ] fp16 support
* Experiments
    - [x] SDF
        - [ ] better SDF calculation (especially for non-watertight meshes)
    - [x] NeRF (cannot converge in 5s of course, but 50s is enough...)
        - [ ] better ray marching strategy.


# Usage

We use the same data format as instant-ngp, e.g., armadillo and fox. 
Please download the data from [instant-ngp](https://github.com/NVlabs/instant-ngp) and put them under `./data`.

```bash
# SDF experiment
bash scripts/run_sdf.sh

# NeRF experiment
bash scripts/run_nerf.sh
```