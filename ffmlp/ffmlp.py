import math
from turtle import backward, forward

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
import atexit

try:
    import _ffmlp as _backend
except ImportError:
    from .backend import _backend

class _ffmlp_forward(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, weights, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, inference=False, calc_grad_inputs=False):
        
        B = inputs.shape[0]

        inputs = inputs.contiguous()
        weights = weights.contiguous()

        # print('[inputs]', torch.any(torch.isnan(inputs)), inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())
        # print('[weights]', torch.any(torch.isnan(weights)), weights.shape, weights.dtype, weights.min().item(), weights.max().item())

        # allocate output
        outputs = torch.empty(B, output_dim, device=inputs.device, dtype=inputs.dtype)

        if not inference:
            forward_buffer = torch.empty(num_layers, B, hidden_dim, device=inputs.device, dtype=inputs.dtype)
            _backend.ffmlp_forward(inputs, weights, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, forward_buffer, outputs)
            ctx.save_for_backward(inputs, weights, outputs, forward_buffer)
            ctx.dims = (input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs)

            # print('[outputs]', torch.any(torch.isnan(outputs)), outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())
            # print('[forward_buffer]', torch.any(torch.isnan(forward_buffer)), forward_buffer.shape, forward_buffer.dtype, forward_buffer.min().item(), forward_buffer.max().item())
        else:
            inference_buffer = torch.empty(B, hidden_dim, device=inputs.device, dtype=inputs.dtype)
            _backend.ffmlp_inference(inputs, weights, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, inference_buffer, outputs)

            # print('[outputs]', torch.any(torch.isnan(outputs)), outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())
            # print('[inference_buffer]', torch.any(torch.isnan(inference_buffer)), inference_buffer.shape, inference_buffer.dtype, inference_buffer.min().item(), inference_buffer.max().item())


        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, output_dim]

        B = grad.shape[0]

        grad = grad.contiguous()

        # print('[grad]', torch.any(torch.isnan(grad)), grad.shape, grad.dtype, grad.min().item(), grad.max().item())
        # print(grad)

        inputs, weights, outputs, forward_buffer = ctx.saved_tensors

        input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs = ctx.dims

        # allocate outputs
        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs) 
        else:
            grad_inputs = torch.zeros(1, device=grad.device, dtype=grad.dtype) # dummy

        grad_weights = torch.zeros_like(weights)
        backward_buffer = torch.zeros(num_layers, B, hidden_dim, device=grad.device, dtype=grad.dtype)

        _backend.ffmlp_backward(grad, inputs, weights, forward_buffer, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs, backward_buffer, grad_inputs, grad_weights)

        # print('[grad_inputs]', grad_inputs.shape, grad_inputs.dtype, grad_inputs.min().item(), grad_inputs.max().item())
        # print('[grad_weights]', grad_weights.shape, grad_weights.dtype, grad_weights.min().item(), grad_weights.max().item())
        # print('[backward_buffer]', backward_buffer.shape, backward_buffer.dtype, backward_buffer.min().item(), backward_buffer.max().item())
        if calc_grad_inputs:
            return grad_inputs, grad_weights, None, None, None, None, None, None, None, None
        else:
            return None, grad_weights, None, None, None, None, None, None, None, None


ffmlp_forward = _ffmlp_forward.apply


def convert_activation(act):
    if act == 'relu': return 0
    elif act == 'exponential': return 1
    elif act == 'sine': return 2
    elif act == 'sigmoid': return 3
    elif act == 'squareplus': return 4
    elif act == 'softplus': return 5
    else: return 6
    

class FFMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation='relu'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = convert_activation(activation)
        self.output_activation = convert_activation('none') # not supported currently

        self.tensorcore_width = 16

        assert hidden_dim in [16, 32, 64, 128, 256], f"FFMLP only support hidden_dim in [16, 32, 64, 128, 256], but got {hidden_dim}"
        assert input_dim > 0 and input_dim % 16 == 0, f"FFMLP input_dim should be 16 * m (m  > 0), but got {input_dim}"
        assert output_dim <= 16, f"FFMLP current only supports output dim <= 16, but got {output_dim}"
        assert num_layers >= 2, f"FFMLP num_layers should be larger than 2 (3 matmuls), but got {num_layers}"
        
        # pad output
        self.padded_output_dim = int(math.ceil(output_dim / 16)) * 16

        # parameters (continuous in memory)
        self.num_parameters = hidden_dim * (input_dim + hidden_dim * (num_layers - 1) + self.padded_output_dim)
        self.weights = nn.Parameter(torch.zeros(self.num_parameters))
        self.reset_parameters()

        # allocate streams
        _backend.allocate_splitk(self.num_layers + 1)

        # register destructor
        #atexit.register(self.cleanup) # how to correctly clean? this gives CUDA Error: cudaEventDestroy(events[i]) failed with error context is destroyed


    def cleanup(self):
        # destroy streams
        _backend.free_splitk()
    

    def __repr__(self):
        return f"FFMLP: input_dim={self.input_dim} output_dim={self.output_dim} hidden_dim={self.hidden_dim} num_layers={self.num_layers} activation={self.activation}"


    def reset_parameters(self):
        torch.manual_seed(42)
        std = math.sqrt(3 / self.hidden_dim)
        self.weights.data.uniform_(-std, std)
    

    def forward(self, inputs):
        # inputs: [B, input_dim]
        # return: [B, outupt_dim]

        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item(), inputs.requires_grad)

        B, C = inputs.shape
        #assert B >= 128 and B % 128 == 0, f"ffmlp batch size must be 128 * m (m > 0), but got {B}."

        # pad input
        pad = 128 - (B % 128)
        if pad > 0:
            inputs = torch.cat([inputs, torch.zeros(pad, C, dtype=inputs.dtype, device=inputs.device)], dim=0)

        outputs = ffmlp_forward(inputs, self.weights, self.input_dim, self.padded_output_dim, self.hidden_dim, self.num_layers, self.activation, self.output_activation, not self.training, inputs.requires_grad)

        # unpad output
        if B != outputs.shape[0] or self.padded_output_dim != self.output_dim:
            outputs = outputs[:B, :self.output_dim]
    
        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs