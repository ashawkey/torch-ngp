# torch-ngp

A pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

**News**: With the CUDA ray marching option for NeRF, we can:
* converge to a reasonable result in **~1min** (50 epochs). 
* render a 1920x1080 image in **~1s**. 

For the LEGO dataset, we can reach **~20FPS** at 800x800 due to efficient voxel pruning.

(Tested on the fox dataset with a TITAN RTX. The speed is still 2-5x slower compared to the original implementation.)

**A GUI for training/visualizing NeRF is also available!**

https://user-images.githubusercontent.com/25863658/155265815-c608254f-2f00-4664-a39d-e00eae51ca59.mp4


# Progress

As the official pytorch extension [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) has been released, the following implementations can be used as modular alternatives. 
The performance and speed of these modules are guaranteed to be on-par, and we support using tinycudann as the backbone by the `--tcnn` flag.
Later development will be focused on reproducing the NeRF inference speed.

* Fully-fused MLP
    - [x] basic pytorch binding of the [original implementation](https://github.com/NVlabs/tiny-cuda-nn)
* HashGrid Encoder
    - [x] basic pytorch CUDA extension
    - [x] fp16 support 
* Experiments
    - SDF
        - [x] baseline
        - [ ] better SDF calculation (especially for non-watertight meshes)
    - NeRF
        - [x] baseline
        - [x] ray marching in CUDA.
* NeRF GUI
    - [x] supports training.
* Misc.
    - [x] improve rendering quality of cuda raymarching!
    - [ ] improve speed (e.g., avoid the `cat` in NeRF forward)
    - [ ] support visualize/supervise normals (add rendering mode option).
    - [x] support blender dataset format.


# Install
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Tested on Ubuntu with torch 1.10 & CUDA 11.3 on TITAN RTX.

Currently, `--ff` only supports GPUs with CUDA architecture `>= 70`.
For GPUs with lower architecture, `--tcnn` can still be used, but the speed will be slower compared to more recent GPUs.

# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

First time running will take some time to compile the CUDA extensions.

```bash
# train with different backbones (with slower pytorch ray marching)
# for the colmap dataset, the default dataset setting `--mode colmap --bound 2 --scale 0.33` is used.
python main_nerf.py data/fox --workspace trial_nerf # fp32 mode
python main_nerf.py data/fox --workspace trial_nerf --fp16 # fp16 mode (pytorch amp)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff # fp16 mode + FFMLP (this repo's implementation)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --tcnn # fp16 mode + official tinycudann's encoder & MLP

# test mode
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --test

# use CUDA to accelerate ray marching (much more faster!)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --cuda_ray # fp16 mode + FFMLP + cuda raymarching

# start a GUI for NeRF training & visualization
# always use with `--fp16 --ff/tcnn --cuda_ray` for an acceptable framerate!
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --cuda_ray --gui

# test mode for GUI
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --cuda_ray --gui --test

# for the blender dataset, you should add `--mode blender --bound 1 --scale 0.8`
# --mode specifies dataset type ('blender' or 'colmap')
# --bound means the scene is assumed to be inside box[-bound, bound]
# --scale adjusts the camera locaction to make sure it falls inside the above bounding box.
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf --fp16 --ff --cuda_ray --mode blender --bound 1 --scale 0.8 
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf --fp16 --ff --cuda_ray --mode blender --bound 1 --scale 0.8 --gui
```

check the `scripts` directory for more provided examples.

# Difference from the original implementation
* Instead of assuming the scene is bounded in the unit box `[0, 1]` and centered at `(0.5, 0.5, 0.5)`, this repo assumes **the scene is bounded in box `[-bound, bound]`, and centered at `(0, 0, 0)`**. Therefore, the functionality of `aabb_scale` is replaced by `bound` here.
* For the hashgrid encoder, this repo only implement the linear interpolation mode.
* For the voxel pruning in ray marching kernels, this repo doesn't implement the multi-scale density grid (check the `mip` keyword), and only use one `128x128x128` grid for simplicity. Instead of updating the grid every 16 steps, we update it every epoch, which may lead to slower first few epochs if using `--cuda_ray`.
* For the blender dataest, the default mode in instant-ngp is to load all data (train/val/test) for training. Instead, we only use the specified split to train in CMD mode for easy evaluation. However, for GUI mode, we follow instant-ngp and use all data to train (check `type='all'` for `NeRFDataset`).

# Update Logs
* 3.9: add fov for gui, 
    * known issue: if the camera is too far-away or fov is too small, the rendered image will be mosaicked. Turning off fp16 can remove the mosaic, so it is most likely related to rays precision.
* 3.1: add type='all' for blender dataset (load train + val + test data), which is the default behavior of instant-ngp.
* 2.28: density_grid now stores density on the voxel center (with randomness), instead of on the grid. This should improve the rendering quality, such as the black strips in the lego scene.
* 2.23: better support for the blender dataset.
* 2.22: add GUI for NeRF training.
* 2.21: add GUI for NeRF visualizing. 
    * known issue: noisy artefacts outside the camera covered region. It is related to `mark_untrained_density_grid` in instant-ngp.
* 2.20: cuda raymarching is finally stable now!
* 2.15: add the official [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) as an alternative backend.    
* 2.10: add cuda_ray, can train/infer faster, but performance is worse currently.
* 2.6: add support for RGBA image.
* 1.30: fixed atomicAdd() to use __half2 in HashGrid Encoder's backward, now the training speed with fp16 is as expected!
* 1.29: 
    * finished an experimental binding of fully-fused MLP.
    * replace SHEncoder with a CUDA implementation.
* 1.26: add fp16 support for HashGrid Encoder (requires CUDA >= 10 and GPU ARCH >= 70 for now...).


# Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/) for the amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp):
    ```
    @misc{tiny-cuda-nn,
        Author = {Thomas M\"uller},
        Year = {2021},
        Note = {https://github.com/nvlabs/tiny-cuda-nn},
        Title = {Tiny {CUDA} Neural Network Framework}
    }

    @article{mueller2022instant,
        title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
        author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
        journal = {arXiv:2201.05989},
        year = {2022},
        month = jan
    }
    ```

* The framework of NeRF is adapted from [nerf_pl](https://github.com/kwea123/nerf_pl):
    ```
    @misc{queianchen_nerf,
        author = {Quei-An, Chen},
        title = {Nerf_pl: a pytorch-lightning implementation of NeRF},
        url = {https://github.com/kwea123/nerf_pl/},
        year = {2020},
    }
    ```
* The NeRF GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
