import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from ffmlp import FFMLP


class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        
        assert self.skips == [], 'FFMLP does not support concatenating inside, please use skips=[].'

        self.encoder, self.in_dim = get_encoder(encoding)

        self.backbone = FFMLP(
            input_dim=self.in_dim, 
            output_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,            
        )

    
    def forward(self, x):
        # x: [B, 3]

        x = self.encoder(x)

        h = self.backbone(x)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h