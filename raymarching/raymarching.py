import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

### generate points (forward only)
# inputs: 
#   rays_o/d: float[N, 3], bound: float
# outputs: 
#   points: float [M, 7], xyzs, dirs, dt
#   rays: int [N, 3], id, offset, num_steps
class _generate_points(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, rays_o, rays_d, bound, density_grid, mean_density, iter_density, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False):
        
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()

        N = rays_o.shape[0] # num rays
        H = density_grid.shape[0] # grid resolution

        M = N * 512 # init max points number in total, hardcoded

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random rays to be ignored.
        if not force_all_rays and mean_count > 0:
            if align > 0:
                mean_count += align - mean_count % align
            M = mean_count
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps

        if step_counter is None:
            step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter

        _backend.generate_points(rays_o, rays_d, density_grid, mean_density, iter_density, bound, N, H, M, xyzs, dirs, deltas, rays, step_counter, perturb) # m is the actually used points number

        #print(step_counter, M)

        # only used at the first epoch.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item() # cause copy to CPU, will slow down a bit
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]

        return xyzs, dirs, deltas, rays

generate_points = _generate_points.apply


### accumulate rays (need backward)
# inputs: sigmas: [M], rgbs: [M, 3], rays: [N, 3], points [M, 7]
# outputs: depth: [N], image: [N, 3]
class _accumulate_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, sigmas, rgbs, deltas, rays, bound, bg_color):
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        deltas = deltas.contiguous()
        rays = rays.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        _backend.accumulate_rays_forward(sigmas, rgbs, deltas, rays, bound, bg_color, M, N, depth, image)

        ctx.save_for_backward(sigmas, rgbs, deltas, rays, image, bg_color)
        ctx.dims = [M, N, bound]

        return depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_depth, grad_image):
        # grad_depth, grad_image: [N, 3]

        grad_image = grad_image.contiguous()

        sigmas, rgbs, deltas, rays, image, bg_color = ctx.saved_tensors
        M, N, bound = ctx.dims
        
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)

        _backend.accumulate_rays_backward(grad_image, sigmas, rgbs, deltas, rays, image, bound, M, N, grad_sigmas, grad_rgbs)

        return grad_sigmas, grad_rgbs, None, None, None, None


accumulate_rays = _accumulate_rays.apply