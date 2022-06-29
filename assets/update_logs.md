## Update logs

* 6.29: add support for HyperNeRF's dataset.
    * we use a simplified pinhole camera model, may introduce bias.
* 6.26: add support for D-NeRF.
    * issue: to enable the `--cuda_ray` in a dynamic scene, we have to record different density grid for different time. This lead to much slower `update_extra_status` and much larger `density_grid` since there is an additional time dimension. Current work arounds: (1) only use 64 time intervals, (2) update it every 100 steps (compared to the 16 steps in static nerf), (3) stop updation after 100 times since the grid should be stable now.
* 6.16: add support for CCNeRF.
* 6.15: fixed a bug in raymarching, improved PSNR. Density thresh is directly applied on sigmas now (removed the empirical scaling factor).
* 6.6: fix gridencoder to always use more accurate float32 inputs (coords), slightly improved performance (matched with tcnn).
* 6.3: implement morton3D, misc improvements.
* 5.29: fix a random bg color issue, add color_space option, better results for blender dataset.
* 5.28: add a background model (set bg_radius > 0), which can suppress noises for real-world 360 datasets.
* 5.21: expose more parameters to control, implement packbits.
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