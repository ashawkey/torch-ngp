import time
import mcubes
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from ffmlp import FFMLP

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
        # TODO: if bound < radius, some rays may not intersect with the bbox.
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
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


# NeRF-SH
class NeRFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 density_grid_size=-1, # density grid size
                 ):
        super().__init__()

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding)

        self.sigma_net = FFMLP(
            input_dim=self.in_dim, 
            output_dim=1 + self.geo_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim + 1 # a manual fixing to make it 32, as done in nerf_network.h#178
        
        self.color_net = FFMLP(
            input_dim=self.in_dim_color, 
            output_dim=3,
            hidden_dim=self.hidden_dim_color,
            num_layers=self.num_layers_color,
        )

        # density grid
        if density_grid_size > 0:
            # buffer is like parameter but never requires_grad
            density_grid = torch.zeros([density_grid_size + 1] * 3) # +1 because we save values at grid
            self.register_buffer('density_grid', density_grid)
            self.mean_density = 0
            self.iter_density = 0
        else:
            self.density_grid = None
    
    def forward(self, x, d, bound):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)

        # sigma
        x = self.encoder(x, size=bound)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color        
        d = self.encoder_dir(d)

        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
    
        sigma = sigma.reshape(*prefix)
        color = color.reshape(*prefix, -1)

        return sigma, color

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)

        x = self.encoder(x, size=bound)
        h = self.sigma_net(x)

        #sigma = torch.exp(torch.clamp(h[..., 0], -15, 15))
        sigma = F.relu(h[..., 0])

        sigma = sigma.reshape(*prefix)

        return sigma

    def run(self, rays_o, rays_d, num_steps, bound, upsample_steps, bg_color):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # sample steps
        near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')

        #print(f'near = {near.min().item()} ~ {near.max().item()}, far = {far.min().item()} ~ {far.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, T]
        z_vals = z_vals.expand((B, N, num_steps)) # [B, N, T]
        z_vals = near + (far - near) * z_vals # [B, N, T], in [near, far]

        # perturb z_vals
        sample_dist = (far - near) / num_steps
        if self.training:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(near, far) # avoid out of bounds pts.

        # generate pts
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, T, 3] -> [B, N, T, 3]
        pts = pts.clamp(-bound, bound) # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

        #print(f'pts {pts.shape} {pts.min().item()} ~ {pts.max().item()}')

        #plot_pointcloud(pts.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        dirs = rays_d.unsqueeze(-2).expand_as(pts)

        sigmas, rgbs = self(pts.reshape(B, -1, 3), dirs.reshape(B, -1, 3), bound=bound)

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
                new_pts = new_pts.clamp(-bound, bound)

            # only forward new points to save computation
            new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)
            new_sigmas, new_rgbs = self(new_pts.reshape(B, -1, 3), new_dirs.reshape(B, -1, 3), bound=bound)
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


    def run_cuda(self, rays_o, rays_d, num_steps, bound, upsample_steps, bg_color):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        if bg_color is None:
            bg_color = torch.ones(3, dtype=rays_o.dtype, device=rays_o.device)

        ### generate points (forward only)
        points, rays = raymarching.generate_points(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, self.training)

        ### call network inference
        # manual pad for ffmlp (slow, should be avoided...)
        n = points.shape[0]
        pad_n = 128 - (n % 128)
        if pad_n > 0:
            points = torch.cat([points, torch.zeros(pad_n, points.shape[1], device=points.device, dtype=points.dtype)], dim=0)

        sigmas, rgbs = self(points[:, :3], points[:, 3:6], bound=bound)

        if pad_n > 0:
            sigmas = sigmas[:n]
            rgbs = rgbs[:n]

        
        ### accumulate rays (need backward)
        # inputs: sigmas: [M], rgbs: [M, 3], offsets: [N+1]
        # outputs: depth: [N], image: [N, 3]
        depth, image = raymarching.accumulate_rays(sigmas, rgbs, points, rays, bound, bg_color)

        depth = depth.reshape(B, N)
        image = image.reshape(B, N, 3)

        return depth, image

    
    def update_density_grid(self, bound, decay=0.95, split_size=128):
        # call before run_cuda, prepare a coarse density grid.

        if self.density_grid is None:
            return 
        
        resolution = self.density_grid.shape[0]

        N = split_size # chunk to avoid OOM
        
        X = torch.linspace(-bound, bound, resolution).split(N)
        Y = torch.linspace(-bound, bound, resolution).split(N)
        Z = torch.linspace(-bound, bound, resolution).split(N)

        tmp_grid = torch.zeros_like(self.density_grid)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                        # manual padding for ffmlp
                        n = pts.shape[0]
                        pad_n = 128 - (n % 128)
                        if pad_n != 0:
                            pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
                        density = self.density(pts.to(tmp_grid.device), bound)[:n].reshape(lx, ly, lz).detach()
                        tmp_grid[xi * N: xi * N + lx, yi * N: yi * N + ly, zi * N: zi * N + lz] = density
        
        # smooth by maxpooling
        tmp_grid = F.pad(tmp_grid, (0, 1, 0, 1, 0, 1))
        tmp_grid = F.max_pool3d(tmp_grid.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=1).squeeze(0).squeeze(0)

        # ema update
        #self.density_grid = tmp_grid
        self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)

        self.mean_density = torch.mean(self.density_grid).item()
        self.iter_density += 1

        # TMP: save mesh for debug
        # vertices, triangles = mcubes.marching_cubes(tmp_grid.detach().cpu().numpy(), 5)
        # vertices = vertices / (resolution - 1.0) * 2 * bound - bound
        # mesh = trimesh.Trimesh(vertices, triangles)
        # mesh.export(f'./tmp/{self.iter_density}.ply')

        print(f'[density grid] iter={self.iter_density} min={self.density_grid.min().item()}, max={self.density_grid.max().item()}, mean={self.mean_density}')


    def render(self, rays_o, rays_d, num_steps, bound, upsample_steps, staged=False, max_ray_batch=4096, bg_color=None, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run if not kwargs['cuda_raymarching'] else self.run_cuda

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)

                    depth_, image_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, bound, upsample_steps, bg_color)
                    
                    depth[b:b+1, head:tail] = depth_
                    image[b:b+1, head:tail] = image_

                    head += max_ray_batch

        else:
            depth, image = _run(rays_o, rays_d, num_steps, bound, upsample_steps, bg_color)

        results = {}
        results['depth'] = depth
        results['rgb'] = image
            
        return results