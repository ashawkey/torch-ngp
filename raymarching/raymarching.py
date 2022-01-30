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
    def forward(ctx, rays_o, rays_d, bound, density_grid, mean_density, iter_density):
        
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()

        N = rays_o.shape[0] # num rays
        H = density_grid.shape[0] # grid resolution

        M = N * 1024 # max points number in total, hardcoded
        
        points = torch.zeros(M, 7, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.zeros(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps
        counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        _backend.generate_points(rays_o, rays_d, density_grid, mean_density, iter_density, bound, N, H, M, points, rays, counter) # m is the actually used points number

        m = counter[0].item()
        points = points[:m]

        print(f"generated points count m = {m} << {M}")

        return points, rays

generate_points = _generate_points.apply


### accumulate rays (need backward)
# inputs: sigmas: [M], rgbs: [M, 3], rays: [N, 3], points [M, 7]
# outputs: depth: [N], image: [N, 3]
class _accumulate_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, sigmas, rgbs, points, rays, bound):
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        points = points.contiguous()
        rays = rays.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        depth = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        _backend.accumulate_rays_forward(sigmas, rgbs, points, rays, bound, M, N, depth, image)

        ctx.save_for_backward(sigmas, rgbs, points, rays, image)
        ctx.dims = [M, N, bound]

        return depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_depth, grad_image):
        # grad_depth, grad_image: [N, 3]

        grad_image = grad_image.contiguous()

        sigmas, rgbs, points, rays, image = ctx.saved_tensors
        M, N, bound = ctx.dims
        
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)

        _backend.accumulate_rays_backward(grad_image, sigmas, rgbs, points, rays, image, bound, M, N, grad_sigmas, grad_rgbs)

        return grad_sigmas, grad_rgbs, None, None, None


accumulate_rays = _accumulate_rays.apply