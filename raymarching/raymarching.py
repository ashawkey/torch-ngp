import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from .backend import _backend

# ----------------------------------------
# utils
# ----------------------------------------

class _near_far_from_aabb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.05):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)

        _backend.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)

        return nears, fars

near_far_from_aabb = _near_far_from_aabb.apply

# ----------------------------------------
# train functions
# ----------------------------------------

class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, density_grid, mean_density, nears, fars, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False, dt_gamma=0):
        ''' march rays to generate points (forward only)
        Args:
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_grid: float, [C, H, H, H]
            mean_density: float, scalar
            nears/fars: float, [N]
            step_counter: int32, (2), used to count the actual number of generated points.
            mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
            perturb: bool
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
        Returns:
            xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs.
            deltas: float, [M,], all generated points' deltas.
            rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 2]] --> points belonging to rays[i, 0]
        '''
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        C = density_grid.shape[0] # grid cascade
        H = density_grid.shape[1] # grid resolution

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
        
        _backend.march_rays_train(rays_o, rays_d, density_grid, mean_density, bound, dt_gamma, N, C, H, M, nears, fars, xyzs, dirs, deltas, rays, step_counter, perturb) # m is the actually used points number

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


class _composite_weights_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, deltas, rays):
        ''' composite rays' weights, according to the ray marching formula.
        Args:
            sigmas: float, [M,]
            deltas: float, [M,]
            rays: int32, [N, 3]
        Returns:
            weights: float, [M,]
        '''
        
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


class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, weights, rgbs, rays):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            weights: float, [M,]
            rays: int32, [N, 3]
        Returns:
            weights_sum: float, [N,], the alpha channel
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        weights = weights.contiguous()
        rgbs = rgbs.contiguous()
        rays = rays.contiguous()

        M = weights.shape[0]
        N = rays.shape[0]

        weights_sum = torch.empty(N, dtype=weights.dtype, device=weights.device)
        image = torch.empty(N, 3, dtype=weights.dtype, device=weights.device)

        _backend.composite_rays_train_forward(weights, rgbs, rays, M, N, weights_sum, image)

        ctx.save_for_backward(weights, rgbs, rays, weights_sum, image)
        ctx.dims = [M, N]

        return weights_sum, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_image):

        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()

        weights, rgbs, rays, weights_sum, image = ctx.saved_tensors
        M, N = ctx.dims
   
        grad_weights = torch.zeros_like(weights)
        grad_rgbs = torch.zeros_like(rgbs)

        _backend.composite_rays_train_backward(grad_weights_sum, grad_image, weights, rgbs, rays, weights_sum, image, M, N, grad_weights, grad_rgbs)

        return grad_weights, grad_rgbs, None, None


composite_rays_train = _composite_rays_train.apply

# ----------------------------------------
# infer functions
# ----------------------------------------

class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_grid, mean_density, near, far, align=-1, perturb=False, dt_gamma=0):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_grid: float, [C, H, H, H]
            mean_density: float, scalar
            nears/fars: float, [N]
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            perturb: bool/int, int > 0 is used as the random seed.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        '''
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        C = density_grid.shape[0] # grid cascade
        H = density_grid.shape[1] # grid resolution
        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device) # 2 vals, one for rgb, one for depth

        _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, dt_gamma, C, H, density_grid, mean_density, near, far, xyzs, dirs, deltas, perturb)

        return xyzs, dirs, deltas

march_rays = _march_rays.apply


class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        _backend.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image)
        return tuple()


composite_rays = _composite_rays.apply


class _compact_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter):
        ''' compact rays, remove dead rays and reallocate alive rays, to accelerate next ray marching.
        Args:
            n_alive: int, number of alive rays
            rays_alive_old: int, [N]
            rays_t_old: float, [N], dead rays are marked by rays_t < 0
            alive_counter: int, [1], used to count remained alive rays.
        In-place Outputs:
            rays_alive: int, [N]
            rays_t: float, [N]
        '''    
        _backend.compact_rays(n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter)
        return tuple()

compact_rays = _compact_rays.apply