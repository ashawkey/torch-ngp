import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.empty(1, device=inputs.device, dtype=inputs.dtype)

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, calc_grad_inputs, dy_dx, gridtype)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs, gridtype)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash'):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution + 1) ** input_dim) # limit max number
            #params_in_level = np.ceil(params_in_level / 8) * 8 # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} params={tuple(self.embeddings.shape)} gridtype={self.gridtype}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs