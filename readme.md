# torch-ngp

A pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).


**A GUI for training/visualizing NeRF is also available!**

https://user-images.githubusercontent.com/25863658/155265815-c608254f-2f00-4664-a39d-e00eae51ca59.mp4


# Install
```bash
git clone --recursive https://github.com/ashawkey/torch-ngp.git
cd torch-ngp
```

### Install with pip
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install with conda
```bash
conda env create -f environment.yml
conda activate torch-ngp
```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Tested environments
* Ubuntu 20 with torch 1.10 & CUDA 11.3 on a TITAN RTX.
* Ubuntu 16 with torch 1.8 & CUDA 10.1 on a V100.
* Windows 10 with torch 1.11 & CUDA 11.3 on a RTX 3070.

Currently, `--ff` only supports GPUs with CUDA architecture `>= 70`.
For GPUs with lower architecture, `--tcnn` can still be used, but the speed will be slower compared to more recent GPUs.


# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

First time running will take some time to compile the CUDA extensions.

```bash
### HashNeRF
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

# preload data into GPU, accelerate training but use more GPU memory.
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --preload

# construct an error_map for each image, and sample rays based on previous training error (experimental)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff --preload --error_map

# one for all: -O means --fp16 --cuda_ray --preload, which usually gives the best results balanced on speed & performance.
python main_nerf.py data/fox --workspace trial_nerf -O

# start a GUI for NeRF training & visualization
# always use with `--fp16 --ff/tcnn --cuda_ray` for an acceptable framerate!
python main_nerf.py data/fox --workspace trial_nerf -O --gui

# test mode for GUI
python main_nerf.py data/fox --workspace trial_nerf -O --gui --test

# for the blender dataset, you should add `--mode blender --bound 1.0 --scale 0.8 --dt_gamma 0`
# --mode specifies dataset type ('blender' or 'colmap')
# --bound means the scene is assumed to be inside box[-bound, bound]
# --scale adjusts the camera locaction to make sure it falls inside the above bounding box. 
# --dt_gamma controls the adaptive ray marching speed, set to 0 turns it off.
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O --mode blender --bound 1.0 --scale 0.8 --dt_gamma 0 
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O --mode blender --bound 1.0 --scale 0.8 --dt_gamma 0 --gui

# for the LLFF dataset, you should first convert it to nerf-compatible format:
python llff2nerf.py data/nerf_llff_data/fern # by default it use full-resolution images, and write `transforms.json` to the folder
python llff2nerf.py data/nerf_llff_data/fern --images images_4 --downscale 4 # if you prefer to use the low-resolution images
# then you can train as a colmap dataset (you'll need to tune the scale & bound if necessary):
python main_nerf.py data/nerf_llff_data/fern --workspace trial_nerf -O
python main_nerf.py data/nerf_llff_data/fern --workspace trial_nerf -O --gui

# for the Tanks&Temples dataset, you should first convert it to nerf-compatible format:
python tanks2nerf.py data/TanksAndTemple/Family # write `trainsforms_{split}.json` for [train, val, test]
# then you can train as a blender dataset (you'll need to tune the scale & bound if necessary)
python main_nerf.py data/TanksAndTemple/Family --workspace trial_nerf_family -O --mode blender --bound 1.0 --scale 0.33 --dt_gamma 0
python main_nerf.py data/TanksAndTemple/Family --workspace trial_nerf_family -O --mode blender --bound 1.0 --scale 0.33 --dt_gamma 0 --gui

# for custom dataset, you should:
# 1. take a video / many photos from different views 
# 2. put the video under a path like ./data/custom/video.mp4 or the images under ./data/custom/images/*.jpg.
# 3. call the preprocess code: (should install ffmpeg and colmap first! refer to the file for more options)
python colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
# 4. it should create the transform.json, and you can train with: (you'll need to try with different scale & bound & dt_gamma to make the object correctly located in the bounding box and render fluently.)
python main_nerf.py data/custom --workspace trial_nerf_custom -O --gui --scale 2.0 --bound 1.0 --dt_gamma 0.02

### SDF
python main_sdf.py data/armadillo.obj --workspace trial_sdf
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16 --ff
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16 --tcnn
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16 --ff --test

### TensoRF
# almost the same as HashNeRF, just replace the main script.
python main_tensoRF.py data/fox --workspace trial_tensoRF -O
python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF -O --mode blender --bound 1.0 --scale 0.8 --dt_gamma 0 

```

check the `scripts` directory for more provided examples.

# Performance Reference
Tested with the default settings on the Lego test dataset. Here the speed refers to the `iterations per second`.
Unfortunately there is still a large performance gap compared to the original implementation.
If you are interested in contributing to this repo, you may start from searching `TODO` in the code.

| Model | PSNR | Train Speed | Test Speed |
| - | - | - | - |
| instant-ngp (paper)               | 36.39  |  -   | -    |
| instant-ngp (`-O`)                | 33.30  |  97  | 7.8  |
| instant-ngp (`-O --error_map`)    | 33.97  |  50  | 7.4  |
| TensoRF (paper)                | 36.46  |  -   | -    |
| TensoRF (`-O`)                 | 34.86  |  51  | 2.8  |
| TensoRF (`-O --error_map`)     | 35.67  |  14  | 2.4  |

# Difference from the original implementation
* Instead of assuming the scene is bounded in the unit box `[0, 1]` and centered at `(0.5, 0.5, 0.5)`, this repo assumes **the scene is bounded in box `[-bound, bound]`, and centered at `(0, 0, 0)`**. Therefore, the functionality of `aabb_scale` is replaced by `bound` here.
* For the hashgrid encoder, this repo only implements the linear interpolation mode.
* For TensoRF, we don't implement regularizations other than L1, and use `trunc_exp` as the density activation instead of `softplus`.


# Update Log
* 4.30: performance improvement (better lr_scheduler).
* 4.25: add Tanks&Temples dataset support.
* 4.18: add some experimental utils for random pose sampling and combined training with CLIP.
* 4.13: add LLFF dataset support.
* 4.13: also implmented tiled grid encoder according to this [issue](https://github.com/NVlabs/instant-ngp/issues/97).
* 4.12: optimized dataloader, add error_map sampling (experimental, will slow down training since will only sample hard rays...)
* 4.10: add Windows support.
* 4.9: use 6D AABB instead of a single `bound` for more flexible rendering. More options in GUI to control the AABB and `dt_gamma` for adaptive ray marching.
* 4.9: implemented multi-res density grid (cascade) and adaptive ray marching. Now the fox renders much faster!
* 4.6: fixed TensorCP hyper-parameters.
* 4.3: add `mark_untrained_grid` to prevent training on out-of-camera regions. Add custom dataset instructions.
* 3.31: better compatibility for lower pytorch versions.
* 3.29: fix training speed for the fox dataset (balanced speed with performance...).
* 3.27: major update. basically improve performance, and support tensoRF model.
* 3.22: reverted from pre-generating rays as it takes too much CPU memory, still the PSNR for Lego can reach ~33 now.
* 3.14: fixed the precision related issue for `fp16` mode, and it renders much better quality. Added PSNR metric for NeRF.
* 3.14: linearly scale `desired_resolution` with `bound` according to https://github.com/ashawkey/torch-ngp/issues/23.
* 3.11: raymarching now supports supervising weights_sum (pixel alpha, or mask) directly, and bg_color is separated from CUDA to make it more flexible. Add an option to preload data into GPU.
* 3.9: add fov for gui.
* 3.1: add type='all' for blender dataset (load train + val + test data), which is the default behavior of instant-ngp.
* 2.28: density_grid now stores density on the voxel center (with randomness), instead of on the grid. This should improve the rendering quality, such as the black strips in the lego scene.
* 2.23: better support for the blender dataset.
* 2.22: add GUI for NeRF training.
* 2.21: add GUI for NeRF visualizing. 
* 2.20: cuda raymarching is finally stable now!
* 2.15: add the official [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) as an alternative backend.    
* 2.10: add cuda_ray, can train/infer faster, but performance is worse currently.
* 2.6: add support for RGBA image.
* 1.30: fixed atomicAdd() to use __half2 in HashGrid Encoder's backward, now the training speed with fp16 is as expected!
* 1.29: finished an experimental binding of fully-fused MLP. replace SHEncoder with a CUDA implementation.
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

* The official TensoRF [implementation](https://github.com/apchenstu/TensoRF):
    ```
    @misc{TensoRF,
        title={TensoRF: Tensorial Radiance Fields},
        author={Anpei Chen and Zexiang Xu and Andreas Geiger and and Jingyi Yu and Hao Su},
        year={2022},
        eprint={2203.09517},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```

* The NeRF GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
