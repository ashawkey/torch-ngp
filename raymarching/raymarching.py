import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

#########################################
### training functions
#########################################

### generate points (forward only)
# inputs: 
#   rays_o/d: float[N, 3], bound: float
# outputs: 
#   points: float [M, 7], xyzs, dirs, dt
#   rays: int [N, 3], id, offset, num_steps
class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, density_grid, mean_density, iter_density, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False):
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays
        H = density_grid.shape[0] # grid resolution

        M = N * 1024 # init max points number in total, hardcoded

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random ignored rays if underestimated.
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

        _backend.march_rays_train(rays_o, rays_d, density_grid, mean_density, iter_density, bound, N, H, M, xyzs, dirs, deltas, rays, step_counter, perturb) # m is the actually used points number

        #print(step_counter, M)

        # only used at the first (few) epochs.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item() # D2H copy
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]

            torch.cuda.empty_cache()

        return xyzs, dirs, deltas, rays

march_rays_train = _march_rays_train.apply

### composite weights
# inputs: sigmas: [M], deltas: [M], rays: [N, 3]
# outputs: weights: [M]
class _composite_weights_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, deltas, rays):
        
        sigmas = sigmas.contiguous()
        deltas = deltas.contiguous()
        rays = rays.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights = torch.zeros(M, dtype=sigmas.dtype, device=sigmas.device)
        
        _backend.composite_weights_forward(sigmas, deltas, rays, M, N, weights)

        ctx.save_for_backward(sigmas, deltas, rays, weights)
        ctx.dims = [M, N]

        return weights

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights):
        
        grad_weights = grad_weights.contiguous()

        sigmas, deltas, rays, weights = ctx.saved_tensors
        M, N = ctx.dims

        grad_sigmas = torch.zeros_like(sigmas)

        _backend.composite_weights_backward(grad_weights, sigmas, deltas, rays, M, N, weights, grad_sigmas)
        return grad_sigmas, None, None


composite_weights_train = _composite_weights_train.apply


### accumulate rays (need backward)
# inputs: sigmas: [M], rgbs: [M, 3], rays: [N, 3]
# outputs: weights_sum: [N], image: [N, 3]
class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, weights, rgbs, rays, bound):
        
        weights = weights.contiguous()
        rgbs = rgbs.contiguous()
        rays = rays.contiguous()

        M = weights.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=weights.dtype, device=weights.device)
        image = torch.empty(N, 3, dtype=weights.dtype, device=weights.device)

        _backend.composite_rays_train_forward(weights, rgbs, rays, bound, M, N, weights_sum, image)

        ctx.save_for_backward(weights, rgbs, rays, weights_sum, image)
        ctx.dims = [M, N, bound]

        return weights_sum, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_image):

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()

        weights, rgbs, rays, weights_sum, image = ctx.saved_tensors
        M, N, bound = ctx.dims
   
        grad_weights = torch.zeros_like(weights)
        grad_rgbs = torch.zeros_like(rgbs)

        _backend.composite_rays_train_backward(grad_weights_sum, grad_image, weights, rgbs, rays, weights_sum, image, bound, M, N, grad_weights, grad_rgbs)

        return grad_weights, grad_rgbs, None, None


composite_rays_train = _composite_rays_train.apply

#########################################
### inference functions
#########################################

### march_rays
# inputs:
#   n_alive: n
#   n_step: int
#   rays_alive: int [n], only the alive IDs in N (n may > n_alive, but we only work on first n_alive)
#   rays_t: float [n], input & output
#   rays_o/d: float [N, 3], all rays
#   bound: float
#   density_grid: float [H, H, H]
#   mean_density: float
#   near/far: float [N]
# outputs:
#   xyzs, dirs, dt: float [n_alive * n_step, 3/3/2], output
class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_grid, mean_density, near, far, align=-1, perturb=False):
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        H = density_grid.shape[0] # grid resolution
        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device) # 2 vals, one for rgb, one for depth

        _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, H, density_grid, mean_density, near, far, xyzs, dirs, deltas, perturb)

        return xyzs, dirs, deltas

march_rays = _march_rays.apply


### composite_rays 
# modify rays_alive to -1 if it is dead.(actual_step < step, indicated by dt <= 0)
# inputs:
#   n_alive: int
#   n_step: int
#   rays_alive: int [n]
#   sigmas, rgbs, deltas: float [n_alive * n_step, 1/3/2]
#   depth, image, weights_sum: float [N, 1/3/1]
class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image):
        _backend.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image)
        return tuple()


composite_rays = _composite_rays.apply

### compact_rays
# inputs:
#   rays_alive_old
#   rays_t_old
# outputs:
#   rays_alive
#   rays_t
class _compact_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter):
        _backend.compact_rays(n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter)
        return tuple()

compact_rays = _compact_rays.apply