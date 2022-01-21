import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

from .backend import _backend

class _hash_encode(Function):
    @staticmethod
    def forward(ctx, inputs, embeddings, offsets, base_resolution, calc_grad_inputs=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        H = base_resolution # base resolution

        outputs = torch.zeros(B,  L * C, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C).to(inputs.device)
        else:
            dy_dx = torch.zeros(1).to(inputs.device)

        _backend.encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, H, calc_grad_inputs, dy_dx)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    def backward(ctx, grad):
        # grad: [B, L * C]

        grad = grad.contiguous()

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1).to(inputs.device)

        _backend.encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, H, calc_grad_inputs, dy_dx, grad_inputs)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None
        else:
            return None, grad_embeddings, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    def __init__(self, D=3, L=16, C=2, base_resolution=16, log2_hashmap_size=19):
        super().__init__()

        self.D = D # coord dims, 2 or 3
        self.L = L # num levels, each level multiply resolution by 2
        self.C = C # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution

        # allocate parameters
        self.offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(L):
            resolution = base_resolution * 2 ** i
            params_in_level = min(self.max_params, (resolution + 1) ** D) # limit max number
            #params_in_level = int(params_in_level / 8) * 8 # make divisible
            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32))
        
        self.n_params = self.offsets[-1] * C

        # parameters
        self.embeddings = nn.Parameter(torch.zeros(offset, C))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: D={self.D} L={self.L} C={self.C} H={self.base_resolution} params={self.embeddings.shape}"
    
    def forward(self, inputs, size=1, calc_grad_inputs=False):
        # inputs: [..., D], normalized real world positions in [-size, size]
        # return: [..., L * C]

        inputs = (inputs + size) / (2 * size) # [0, 1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.D)

        outputs = hash_encode(inputs, self.embeddings, self.offsets, self.base_resolution, calc_grad_inputs)
        outputs = outputs.reshape(prefix_shape + [self.L * self.C])

        return outputs