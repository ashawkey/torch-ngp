# torch-ngp

This repository contains:
* A pytorch implementation of the SDF and NeRF part (grid encoder, density grid ray sampler) in [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).
* A pytorch implementation of [TensoRF](https://github.com/apchenstu/TensoRF), as described in [_TensoRF: Tensorial Radiance Fields_](https://arxiv.org/abs/2203.09517), adapted to instant-ngp's NeRF framework.
* A pytorch implementation of [CCNeRF](https://github.com/ashawkey/CCNeRF), as described in [_Compressible-composable NeRF via Rank-residual Decomposition_](https://arxiv.org/abs/2205.14870).
* [New!] An implementation of [D-NeRF](https://github.com/albertpumarola/D-NeRF) adapted to instant-ngp's framework, as described in [_D-NeRF: Neural Radiance Fields for Dynamic Scenes_](https://openaccess.thecvf.com/content/CVPR2021/papers/Pumarola_D-NeRF_Neural_Radiance_Fields_for_Dynamic_Scenes_CVPR_2021_paper.pdf).
* Some experimental features in the NeRF framework (e.g., text-guided NeRF editig similar to [CLIP-NeRF](https://arxiv.org/abs/2112.05139)).
* A GUI for training/visualizing NeRF!

### [Gallery](assets/gallery.md) | [Update Logs](assets/update_logs.md)

Instant-ngp interactive training/rendering on lego:

https://user-images.githubusercontent.com/25863658/176174011-e7b7c4ab-9b6f-4f65-9952-7eceafe609b7.mp4

Also the first interactive deformable-nerf implementation:

https://user-images.githubusercontent.com/25863658/175821784-63ba79f6-29be-47b5-b3fc-dab5282fce7a.mp4


### Other related projects

* [ngp_pl](https://github.com/kwea123/ngp_pl): PyTorch+CUDA trained with pytorch-lightning.

* [JNeRF](https://github.com/Jittor/JNeRF): An NeRF benchmark based on Jittor.

* [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch): A pure PyTorch implementation.

* [dreamfields-torch](https://github.com/ashawkey/dreamfields-torch): PyTorch+CUDA implementation of [_Zero-Shot Text-Guided Object Generation with Dream Fields_](https://arxiv.org/abs/2112.01455) based on this repository.

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

We also support self-captured dataset and converting other formats (e.g., LLFF, Tanks&Temples, Mip-NeRF 360) to the nerf-compatible format, with details in the following code block.

<details>
  <summary> Supported datasets </summary>

  * [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 

  * [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip): [[conversion script]](./scripts/tanks2nerf.py)

  * [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7): [[conversion script]](./scripts/llff2nerf.py)

  * [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip): [[conversion script]](./scripts/llff2nerf.py)

  * (dynamic) [D-NeRF](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0)

  * (dynamic) [Hyper-NeRF](https://github.com/google/hypernerf/releases/tag/v0.1): [[conversion script]](./scripts/hyper2nerf.py)

</details>

First time running will take some time to compile the CUDA extensions.

```bash
### Instant-ngp NeRF
# train with different backbones (with slower pytorch ray marching)
# for the colmap dataset, the default dataset setting `--bound 2 --scale 0.33` is used.
python main_nerf.py data/fox --workspace trial_nerf # fp32 mode
python main_nerf.py data/fox --workspace trial_nerf --fp16 # fp16 mode (pytorch amp)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --ff # fp16 mode + FFMLP (this repo's implementation)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --tcnn # fp16 mode + official tinycudann's encoder & MLP

# use CUDA to accelerate ray marching (much more faster!)
python main_nerf.py data/fox --workspace trial_nerf --fp16 --cuda_ray # fp16 mode + cuda raymarching

# preload data into GPU, accelerate training but use more GPU memory.
python main_nerf.py data/fox --workspace trial_nerf --fp16 --preload

# one for all: -O means --fp16 --cuda_ray --preload, which usually gives the best results balanced on speed & performance.
python main_nerf.py data/fox --workspace trial_nerf -O

# test mode
python main_nerf.py data/fox --workspace trial_nerf -O --test

# construct an error_map for each image, and sample rays based on the training error (slow down training but get better performance with the same number of training steps)
python main_nerf.py data/fox --workspace trial_nerf -O --error_map

# use a background model (e.g., a sphere with radius = 32), can supress noises for real-world 360 dataset
python main_nerf.py data/firekeeper --workspace trial_nerf -O --bg_radius 32

# start a GUI for NeRF training & visualization
# always use with `--fp16 --cuda_ray` for an acceptable framerate!
python main_nerf.py data/fox --workspace trial_nerf -O --gui

# test mode for GUI
python main_nerf.py data/fox --workspace trial_nerf -O --gui --test

# for the blender dataset, you should add `--bound 1.0 --scale 0.8 --dt_gamma 0`
# --bound means the scene is assumed to be inside box[-bound, bound]
# --scale adjusts the camera locaction to make sure it falls inside the above bounding box. 
# --dt_gamma controls the adaptive ray marching speed, set to 0 turns it off.
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O --bound 1.0 --scale 0.8 --dt_gamma 0
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O --bound 1.0 --scale 0.8 --dt_gamma 0 --gui

# for the LLFF dataset, you should first convert it to nerf-compatible format:
python scripts/llff2nerf.py data/nerf_llff_data/fern # by default it use full-resolution images, and write `transforms.json` to the folder
python scripts/llff2nerf.py data/nerf_llff_data/fern --images images_4 --downscale 4 # if you prefer to use the low-resolution images
# then you can train as a colmap dataset (you'll need to tune the scale & bound if necessary):
python main_nerf.py data/nerf_llff_data/fern --workspace trial_nerf -O
python main_nerf.py data/nerf_llff_data/fern --workspace trial_nerf -O --gui

# for the Tanks&Temples dataset, you should first convert it to nerf-compatible format:
python scripts/tanks2nerf.py data/TanksAndTemple/Family # write `trainsforms_{split}.json` for [train, val, test]
# then you can train as a blender dataset (you'll need to tune the scale & bound if necessary)
python main_nerf.py data/TanksAndTemple/Family --workspace trial_nerf_family -O --bound 1.0 --scale 0.33 --dt_gamma 0
python main_nerf.py data/TanksAndTemple/Family --workspace trial_nerf_family -O --bound 1.0 --scale 0.33 --dt_gamma 0 --gui

# for custom dataset, you should:
# 1. take a video / many photos from different views 
# 2. put the video under a path like ./data/custom/video.mp4 or the images under ./data/custom/images/*.jpg.
# 3. call the preprocess code: (should install ffmpeg and colmap first! refer to the file for more options)
python scripts/colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python scripts/colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
python scripts/colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap --dynamic # if the scene is dynamic (for D-NeRF settings), add the time for each frame.
# 4. it should create the transform.json, and you can train with: (you'll need to try with different scale & bound & dt_gamma to make the object correctly located in the bounding box and render fluently.)
python main_nerf.py data/custom --workspace trial_nerf_custom -O --gui --scale 2.0 --bound 1.0 --dt_gamma 0.02

### Instant-ngp SDF
python main_sdf.py data/armadillo.obj --workspace trial_sdf
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16 --ff
python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16 --tcnn

python main_sdf.py data/armadillo.obj --workspace trial_sdf --fp16 --test

### TensoRF
# almost the same as Instant-ngp NeRF, just replace the main script.
python main_tensoRF.py data/fox --workspace trial_tensoRF -O
python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF -O --bound 1.0 --scale 0.8 --dt_gamma 0 

### CCNeRF
# training on single objects, turn on --error_map for better quality.
python main_CCNeRF.py data/nerf_synthetic/chair --workspace trial_cc_chair -O --bound 1.0 --scale 0.67 --dt_gamma 0 --error_map
python main_CCNeRF.py data/nerf_synthetic/ficus --workspace trial_cc_ficus -O --bound 1.0 --scale 0.67 --dt_gamma 0 --error_map
python main_CCNeRF.py data/nerf_synthetic/hotdog --workspace trial_cc_hotdog -O --bound 1.0 --scale 0.67 --dt_gamma 0 --error_map
# compose, use a larger bound and more samples per ray for better quality.
python main_CCNeRF.py data/nerf_synthetic/hotdog --workspace trial_cc_hotdog -O --bound 2.0 --scale 0.67 --dt_gamma 0 --max_steps 2048 --test --compose
# compose + gui, only about 1 FPS without dynamic resolution... just for quick verification of composition results.
python main_CCNeRF.py data/nerf_synthetic/hotdog --workspace trial_cc_hotdog -O --bound 2.0 --scale 0.67 --dt_gamma 0 --test --compose --gui

### D-NeRF
# almost the same as Instant-ngp NeRF, just replace the main script.
python main_dnerf.py data/dnerf/jumpingjacks --workspace trial_dnerf_jumpingjacks -O --bound 1.0 --scale 0.8 --dt_gamma 0
python main_dnerf.py data/dnerf/jumpingjacks --workspace trial_dnerf_jumpingjacks -O --bound 1.0 --scale 0.8 --dt_gamma 0 --gui
# for the hypernerf dataset, first convert it into nerf-compatible format:
python scripts/hyper2nerf.py data/split-cookie --downscale 2 # will generate transforms*.json
python main_dnerf.py data/split-cookie/ --workspace trial_dnerf_cookies -O --bound 1 --scale 0.3 --dt_gamma 0
```

check the `scripts` directory for more provided examples.

# Performance Reference

Tested with the default settings on the Lego dataset.
Here the speed refers to the `iterations per second` on a V100.

| Model | Split | PSNR | Train Speed | Test Speed |
| - | - | - | - | - |
| instant-ngp (paper)            | trainval?            | 36.39  |  -   | -    |
| instant-ngp (`-O`)             | train (30K steps)    | 34.15  |  97  | 7.8  |
| instant-ngp (`-O --error_map`) | train (30K steps)    | 34.88  |  50  | 7.8  |
| instant-ngp (`-O`)             | trainval (40k steps) | 35.22  |  97  | 7.8  |
| instant-ngp (`-O --error_map`) | trainval (40k steps) | 36.00  |  50  | 7.8  |
| TensoRF (paper)                | train (30K steps)    | 36.46  |  -   | -    |
| TensoRF (`-O`)                 | train (30K steps)    | 35.05  |  51  | 2.8  |
| TensoRF (`-O --error_map`)     | train (30K steps)    | 35.84  |  14  | 2.8  |

# Tips

**Q**: How to choose the network backbone? 

**A**: The `-O` flag which uses pytorch's native mixed precision is suitable for most cases. I don't find very significant improvement for `--tcnn` and `--ff`, and they require extra building. Also, some new features may only be available for the default `-O` mode.

**Q**: CUDA Out Of Memory for my dataset.

**A**: You could try to turn off `--preload` which loads all images in to GPU for acceleration (if use `-O`, change it to `--fp16 --cuda_ray`). Another solution is to manually set `downscale` in `NeRFDataset` to lower the image resolution.

**Q**: How to adjust `bound` and `scale`? 

**A**: You could start with a large `bound` (e.g., 16) or a small `scale` (e.g., 0.3) to make sure the object falls into the bounding box. The GUI mode can be used to interactively shrink the `bound` to find the suitable value. Uncommenting [this line](https://github.com/ashawkey/torch-ngp/blob/main/nerf/provider.py#L219) will visualize the camera poses, and some good examples can be found in [this issue](https://github.com/ashawkey/torch-ngp/issues/59).

**Q**: Noisy novel views for realistic datasets.

**A**: You could try setting `bg_radius` to a large value, e.g., 32. It trains an extra environment map to model the background in realistic photos. A larger `bound` will also help.
An example for `bg_radius` in the [firekeeper](https://drive.google.com/file/d/19C0K6_crJ5A9ftHijUmJysxmY-G4DMzq/view?usp=sharing) dataset:
![bg_model](./assets/bg_model.jpg)


# Difference from the original implementation

* Instead of assuming the scene is bounded in the unit box `[0, 1]` and centered at `(0.5, 0.5, 0.5)`, this repo assumes **the scene is bounded in box `[-bound, bound]`, and centered at `(0, 0, 0)`**. Therefore, the functionality of `aabb_scale` is replaced by `bound` here.
* For the hashgrid encoder, this repo only implements the linear interpolation mode.
* For TensoRF, we don't implement regularizations other than L1, and use `trunc_exp` as the density activation instead of `softplus`. The alpha mask pruning is replaced by the density grid sampler from instant-ngp, which shares the same logic for acceleration.


# Citation

If you find this work useful, a citation will be appreciated via:
```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}

@article{tang2022compressible,
    title = {Compressible-composable NeRF via Rank-residual Decomposition},
    author = {Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
    journal = {arXiv preprint arXiv:2205.14870},
    year = {2022}
}
```

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
    @article{TensoRF,
      title={TensoRF: Tensorial Radiance Fields},
      author={Chen, Anpei and Xu, Zexiang and Geiger, Andreas and Yu, Jingyi and Su, Hao},
      journal={arXiv preprint arXiv:2203.09517},
      year={2022}
    }
    ```

* The NeRF GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
