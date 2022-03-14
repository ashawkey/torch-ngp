import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=0.05)

    return near, far


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 ):
        super().__init__()

        self.bound = bound

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([128] * 3)
            self.register_buffer('density_grid', density_grid)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(64, 2, dtype=torch.int32) # 64 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # sample steps
        near, far = near_far_from_bound(rays_o, rays_d, self.bound, type='cube')

        #print(f'near = {near.min().item()} ~ {near.max().item()}, far = {far.min().item()} ~ {far.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, T]
        z_vals = z_vals.expand((B, N, num_steps)) # [B, N, T]
        z_vals = near + (far - near) * z_vals # [B, N, T], in [near, far]

        # perturb z_vals
        sample_dist = (far - near) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(near, far) # avoid out of bounds pts.

        # generate pts
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, T, 3] -> [B, N, T, 3]
        pts = pts.clamp(-self.bound, self.bound) # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

        #print(f'pts {pts.shape} {pts.min().item()} ~ {pts.max().item()}')

        #plot_pointcloud(pts.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        dirs = rays_d.unsqueeze(-2).expand_as(pts)

        sigmas, rgbs = self(pts.reshape(B, -1, 3), dirs.reshape(B, -1, 3))

        rgbs = rgbs.reshape(B, N, num_steps, 3) # [B, N, T, 3]
        sigmas = sigmas.reshape(B, N, num_steps) # [B, N, T]

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # [B, N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[:, :, :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * sigmas) # [B, N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-15], dim=-1) # [B, N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :, :-1] # [B, N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[:, :, :-1] + 0.5 * deltas[:, :, :-1]) # [B, N, T-1]
                new_z_vals = sample_pdf(z_vals_mid.reshape(B*N, -1), weights.reshape(B*N, -1)[:, 1:-1], upsample_steps, det=not self.training).detach() # [BN, t]
                new_z_vals = new_z_vals.reshape(B, N, upsample_steps)

                new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, t, 3] -> [B, N, t, 3]
                new_pts = new_pts.clamp(-self.bound, self.bound)

            # only forward new points to save computation
            new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)
            new_sigmas, new_rgbs = self(new_pts.reshape(B, -1, 3), new_dirs.reshape(B, -1, 3))
            new_rgbs = new_rgbs.reshape(B, N, upsample_steps, 3) # [B, N, t, 3]
            new_sigmas = new_sigmas.reshape(B, N, upsample_steps) # [B, N, t]

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=-1) # [B, N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=-1)

            sigmas = torch.cat([sigmas, new_sigmas], dim=-1) # [B, N, T+t]
            sigmas = torch.gather(sigmas, dim=-1, index=z_index)

            rgbs = torch.cat([rgbs, new_rgbs], dim=-2) # [B, N, T+t, 3]
            rgbs = torch.gather(rgbs, dim=-2, index=z_index.unsqueeze(-1).expand_as(rgbs))

        ### render core
        deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # [B, N, T-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[:, :, :1])], dim=-1)

        alphas = 1 - torch.exp(-deltas * sigmas) # [B, N, T]
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-15], dim=-1) # [B, N, T+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :, :-1] # [B, N, T]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [B, N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [B, N, 3], in [0, 1]

        # mix background color
        if bg_color is None:
            bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        return depth, image


    def run_cuda(self, rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if bg_color is None:
            bg_color = 1

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 64]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, perturb, 128, False)
            sigmas, rgbs = self(xyzs, dirs)
            weights_sum, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, self.bound)

            # composite bg (shade_kernel_nerf)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = None # currently training do not requires depth

        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(B * N, dtype=dtype, device=device)
            depth = torch.zeros(B * N, dtype=dtype, device=device)
            image = torch.zeros(B * N, 3, dtype=dtype, device=device)
            
            n_alive = B * N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            # pre-calculate near far
            near, far = near_far_from_bound(rays_o, rays_d, self.bound, type='cube')
            near = near.view(B * N)
            far = far.view(B * N)

            step = 0
            i = 0
            while step < 1024: # max step

                # count alive rays 
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = near
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2], rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item() # must invoke D2H copy here
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(B * N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, self.bound, self.density_grid, self.mean_density, near, far, 128, perturb)
                sigmas, rgbs = self(xyzs, dirs)
                raymarching.composite_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], sigmas, rgbs, deltas, weights_sum, depth, image)

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}')

                step += n_step
                i += 1

            # composite bg & rectify depth (shade_kernel_nerf)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - near, min=0) / (far - near)


        image = image.reshape(B, N, 3)
        if depth is not None:
            depth = depth.reshape(B, N)

        return depth, image

    
    def update_extra_state(self, decay=0.95):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid
        resolution = self.density_grid.shape[0]

        half_grid_size = self.bound / resolution
        
        X = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(128)
        Y = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(128)
        Z = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(128)

        tmp_grid = torch.zeros_like(self.density_grid)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        # construct points
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                        # add noise in [-hgs, hgs]
                        pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size
                        # manual padding for ffmlp
                        n = pts.shape[0]
                        pad_n = 128 - (n % 128)
                        if pad_n != 0:
                            pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
                        # query density
                        density = self.density(pts.to(tmp_grid.device))[:n].reshape(lx, ly, lz).detach()
                        tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density
        
        # ema update
        self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)
        self.mean_density = torch.mean(self.density_grid).item()
        self.iter_density += 1

        ### update step counter
        total_step = min(64, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f} | [step counter] mean={self.mean_count}')


    def render(self, rays_o, rays_d, num_steps=128, upsample_steps=128, staged=False, max_ray_batch=4096, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    depth_, image_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, upsample_steps, bg_color, perturb)
                    depth[b:b+1, head:tail] = depth_
                    image[b:b+1, head:tail] = image_
                    head += max_ray_batch
        else:
            depth, image = _run(rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb)

        results = {}
        results['depth'] = depth
        results['rgb'] = image
            
        return results