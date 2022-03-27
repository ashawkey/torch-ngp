import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from nerf.renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 resolution=[128] * 3,
                 sigma_rank=[16] * 3,
                 color_rank=[48] * 3,
                 color_feat_dim=27,
                 num_layers=3,
                 hidden_dim=128,
                 bound=1,
                 cuda_ray=False,
                 ):
        super().__init__(bound, cuda_ray)

        self.resolution = resolution

        # vector-matrix decomposition
        self.sigma_rank = sigma_rank
        self.color_rank = color_rank
        self.color_feat_dim = color_feat_dim

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.sigma_mat, self.sigma_vec = self.init_one_svd(self.sigma_rank, self.resolution)
        self.color_mat, self.color_vec = self.init_one_svd(self.color_rank, self.resolution)
        self.basis_mat = nn.Linear(sum(self.color_rank), self.color_feat_dim, bias=False)

        # render module (default to freq feat + freq dir)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, enc_dim = get_encoder('frequency', input_dim=color_feat_dim, multires=2)
        self.encoder_dir, enc_dim_dir = get_encoder('frequency', input_dim=3, multires=2)

        self.in_dim = enc_dim + enc_dim_dir

        color_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3 # rgb
            else:
                out_dim = self.hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)


    def init_one_svd(self, n_component, resolution, scale=0.1):

        mat, vec = [], []

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]
            mat.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], resolution[mat_id_1], resolution[mat_id_0])))) # [1, R, H, W]
            vec.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], resolution[vec_id], 1)))) # [1, R, D, 1] (fake 2d to use grid_sample)

        return torch.nn.ParameterList(mat), torch.nn.ParameterList(vec)


    def get_sigma_feat(self, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        sigma_feat = torch.zeros([N,], device=x.device)

        for i in range(len(self.sigma_mat)):
            mat_feat = F.grid_sample(self.sigma_mat[i], mat_coord[[i]], align_corners=True).view(-1, N) # [1, R, N, 1] --> [R, N]
            vec_feat = F.grid_sample(self.sigma_vec[i], vec_coord[[i]], align_corners=True).view(-1, N) # [R, N]
            sigma_feat = sigma_feat + torch.sum(mat_feat * vec_feat, dim=0)

        return sigma_feat


    def get_color_feat(self, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        mat_feat, vec_feat = [], []

        for i in range(len(self.color_mat)):
            mat_feat.append(F.grid_sample(self.color_mat[i], mat_coord[[i]], align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
            vec_feat.append(F.grid_sample(self.color_vec[i], vec_coord[[i]], align_corners=True).view(-1, N)) # [R, N]
        
        mat_feat = torch.cat(mat_feat, dim=0) # [3 * R, N]
        vec_feat = torch.cat(vec_feat, dim=0) # [3 * R, N]

        color_feat = self.basis_mat((mat_feat * vec_feat).T) # [N, 3R] --> [N, color_feat_dim]

        return color_feat
    
    
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # normalize to [-1, 1]
        x = x / self.bound

        # sigma
        sigma_feat = self.get_sigma_feat(x)
        sigma = F.relu(sigma_feat, inplace=True)

        # rgb
        color_feat = self.get_color_feat(x)
        enc_color_feat = self.encoder(color_feat)
        enc_d = self.encoder_dir(d)

        h = torch.cat([enc_color_feat, enc_d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return sigma, rgb


    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        # normalize to [-1, 1]
        x = x / self.bound

        sigma_feat = self.get_sigma_feat(x)
        sigma = F.relu(sigma_feat, inplace=True)

        return {
            'sigma': sigma,
        }

    # allow masked inference
    def color(self, x, d, mask=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        # normalize to [-1, 1]
        x = x / self.bound

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]

        color_feat = self.get_color_feat(x)
        color_feat = self.encoder(color_feat)
        d = self.encoder_dir(d)

        h = torch.cat([color_feat, d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs


    # L1 penalty for loss
    def density_loss(self):
        loss = 0
        for i in range(len(self.sigma_mat)):
            loss = loss + torch.mean(torch.abs(self.sigma_mat[i])) + torch.mean(torch.abs(self.sigma_vec[i]))
        return loss
    
    # upsample utils
    @torch.no_grad()
    def upsample_params(self, mat, vec, resolution):

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]
            mat[i] = torch.nn.Parameter(F.interpolate(mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=True))
            vec[i] = torch.nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))

        return mat, vec

    @torch.no_grad()
    def upsample_model(self, resolution):
        self.sigma_mat, self.sigma_vec = self.upsample_params(self.sigma_mat, self.sigma_vec, resolution)
        self.color_mat, self.color_vec = self.upsample_params(self.color_mat, self.color_vec, resolution)
        self.resolution = resolution

    # optimizer utils
    def get_params(self, lr1, lr2):
        return [
            {'params': self.sigma_mat, 'lr': lr1}, 
            {'params': self.sigma_vec, 'lr': lr1},
            {'params': self.color_mat, 'lr': lr1}, 
            {'params': self.color_vec, 'lr': lr1},
            {'params': self.basis_mat.parameters(), 'lr': lr2},
            {'params': self.color_net.parameters(), 'lr': lr2},
        ]
        