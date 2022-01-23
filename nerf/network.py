import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder

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

def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    import trimesh
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
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 ):
        super().__init__()

        
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def forward(self, x, d, bound):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, size=bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        
        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        x = self.encoder(x, size=bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = h[..., 0]

        return sigma

    def run(self, rays_o, rays_d, num_steps, bound, upsample_steps):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # sample steps
        near = rays_o.norm(dim=-1, keepdim=True) - bound # [B, N, 1]
        far = near + 2 * bound

        #print(f'near = {near.min().item()} ~ {near.max().item()}, far = {far.min().item()} ~ {far.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, T]
        z_vals = z_vals.expand((B, N, num_steps)) # [B, N, T]
        z_vals = near + (far - near) * z_vals # [B, N, T], in [near, far]

        # pertube z_vals
        sample_dist = (far - near) / num_steps
        if self.training:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate pts
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, T, 3] -> [B, N, T, 3]

        #print(f'pts {pts.shape} {pts.min().item()} ~ {pts.max().item()}')

        #plot_pointcloud(pts.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        rays_d_ = rays_d.unsqueeze(-2).expand_as(pts)
        sigmas, rgbs = self(pts.reshape(B, -1, 3), rays_d_.reshape(B, -1, 3), bound=bound)
        rgbs = rgbs.reshape(B, N, num_steps, 3) # [B, N, T, 3]
        sigmas = sigmas.reshape(B, N, num_steps) # [B, N, T]

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # [B, N, T-1]
                deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :, :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * (F.relu(sigmas))) # [B, N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-7], dim=-1) # [B, N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :, :-1] # [B, N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[:, :, :-1] + 0.5 * deltas[:, :, :-1]) # [B, N, T-1]
                new_z_vals = sample_pdf(z_vals_mid.reshape(B*N, -1), weights.reshape(B*N, -1)[:, 1:-1], upsample_steps, det=not self.training).detach() # [BN, t]
                new_z_vals = new_z_vals.reshape(B, N, upsample_steps)

                new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, t, 3] -> [B, N, t, 3]

            # only forward new points to save computation
            new_rays_d_ = rays_d.unsqueeze(-2).expand_as(new_pts)
            new_sigmas, new_rgbs = self(new_pts.reshape(B, -1, 3), new_rays_d_.reshape(B, -1, 3), bound=bound)
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
        deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :, :1])], dim=-1)

        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas))) # [B, N, T]
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-7], dim=-1) # [B, N, T+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :, :-1] # [B, N, T]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [B, N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)
        depth = (torch.sum(weights * ori_z_vals, dim=-1) + 1e-7) / (weights_sum + 1e-7) # [B, N], in [0, 1] (infinite is 1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [B, N, 3], in [0, 1]
        #image = image + (1 - weights_sum).unsqueeze(-1) # white background (infinite depth)

        return depth, image

    def render(self, rays_o, rays_d, num_steps, bound, upsample_steps, staged=False, max_ray_batch=256000, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged:
            depth = torch.zeros((B, N), device=device)
            image = torch.zeros((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)

                    depth_, image_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, bound, upsample_steps)
                    
                    depth[b:b+1, head:tail] = depth_
                    image[b:b+1, head:tail] = image_

                    head += max_ray_batch

        else:
            depth, image = self.run(rays_o, rays_d, num_steps, bound, upsample_steps)

        results = {}
        results['depth'] = depth
        results['rgb'] = image
            
        return results