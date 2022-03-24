import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from ffmlp import FFMLP

from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 cuda_ray=False,
                 ):
        super().__init__(bound, cuda_ray)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

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
    
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color        
        d = self.encoder_dir(d)

        # TODO: avoid this cat op... 
        # should pre-allocate output (col-major!), then inplace write from ffmlp & shencoder, finally transpose to row-major.
        # this is impossible...
        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return sigma, rgb

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)

        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs = torch.zeros(np.prod(prefix), 3, dtype=h.dtype, device=h.device) # [N, 3]
            rgbs[mask] = h
        else:
            rgbs = h

        return rgbs
