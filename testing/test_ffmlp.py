from matplotlib.animation import AVConvBase
import torch
import torch.nn as nn
import torch.nn.functional as F

from ffmlp import FFMLP
import math

import tinycudann as tcnn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(input_dim, hidden_dim, bias=False))
        for i in range(num_layers - 1):
            self.net.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.net.append(nn.Linear(hidden_dim, output_dim, bias=False))

        self.reset_parameters()
    
    def reset_parameters(self):
        torch.manual_seed(42)
        for p in self.parameters():
            #nn.init.constant_(p.data, 1)
            std = math.sqrt(3 / self.hidden_dim)
            p.data.uniform_(-std, std)
            #torch.manual_seed(42)
            #nn.init.uniform_(p.data, 0, 1)
            #nn.init.eye_(p.data)

    
    def forward(self, x):
        for i in range(self.num_layers + 1):
            x = self.net[i](x)
            if i != self.num_layers:
                x = self.activation(x)
        return x

# ##################################
# # Functionality
# ##################################

# BATCH_SIZE = 1280000 # 1048576 # 128 # the least batch to lauch a full block !
# INPUT_DIM = 16 # 16 # != (16 * m) has bug...
# OUTPUT_DIM = 1 # 16 # > 16 still has bug...
# HIDDEN_DIM = 64 # 16
# NUM_LAYERS = 3 # 2


# net0 = FFMLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS).cuda()
# net1 = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS).cuda()

# # print(net0.weights)
# # print(net1.net[0].weight)

# for _ in range(5):

#     x0 = torch.randn(BATCH_SIZE, INPUT_DIM).cuda() * 1
#     x1 = x0.detach().clone()
#     x0.requires_grad_(True)
#     x1.requires_grad_(True)

#     # print('===== x =====')
#     # print(x0)
#     # print(x1)

#     with torch.cuda.amp.autocast(enabled=True):
#         y1 = net1(x1)
#         y0 = net0(x0)


#     print('===== y1 =====')
#     print(y1)

#     print('===== y0 =====')
#     print(y0)

#     (y1.sum() * 1).backward()
#     print('===== grad w1 =====')
#     print(net1.net[0].weight.grad.dtype, torch.cat([net1.net[0].weight.grad.view(-1), net1.net[1].weight.grad.view(-1), net1.net[2].weight.grad.view(-1)], dim=0))
#     print(x1.grad.dtype, x1.grad)

#     (y0.sum() * 1).backward()
#     print('===== grad w0 =====')
#     print(net0.weights.grad.dtype, net0.weights.grad)
#     print(x0.grad.dtype, x0.grad)



# ##################################
# # Speed
# ##################################

BATCH_SIZE = 2**21
INPUT_DIM = 16
OUTPUT_DIM = 16
HIDDEN_DIM = 64
NUM_LAYERS = 2

net0 = FFMLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS).cuda()
net1 = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS).cuda()
net2 = tcnn.Network(n_input_dims=INPUT_DIM, n_output_dims=OUTPUT_DIM, network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": HIDDEN_DIM,
                    "n_hidden_layers": NUM_LAYERS,
                })

x = torch.rand(BATCH_SIZE, INPUT_DIM).cuda() * 10
x1 = x.detach().clone()
x2 = x.detach().clone()
x3 = x.detach().clone()



#with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
y2 = net1(x2)
ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'pytorch MLP (fp32 train) = {curr_time}')

starter.record()
y2.sum().backward()
ender.record()
torch.cuda.synchronize()
curr_time = starter.elapsed_time(ender)
print(f'pytorch MLP (fp32 back) = {curr_time}')

#print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

with torch.cuda.amp.autocast(enabled=True):

    #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:
        starter.record()
        y0 = net0(x)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'FFMLP (forward) = {curr_time}')

        starter.record()
        y0.sum().backward()
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'FFMLP (backward) = {curr_time}')
        
    #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:
        starter.record()
        y1 = net1(x1)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'pytorch MLP (forward) = {curr_time}')

        starter.record()
        y1.sum().backward()
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'pytorch MLP (backward) = {curr_time}')
    #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:
        starter.record()
        y3 = net2(x3)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'TCNN (forward) = {curr_time}')

        starter.record()
        y3.sum().backward()
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'TCNN (backward) = {curr_time}')
    #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

with torch.no_grad():
    
    starter.record()
    y1 = net1(x)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    print(f'pytorch MLP (fp32 infer) = {curr_time}')

    with torch.cuda.amp.autocast(enabled=True):
        
        
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:

            starter.record()
            y0 = net0(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            print(f'FFMLP (infer) = {curr_time}')

        #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:

            starter.record()
            y1 = net1(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            print(f'pytorch MLP (infer) = {curr_time}')

        #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as p:

            starter.record()
            y2 = net2(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            print(f'TCNN (infer) = {curr_time}')

        #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


# print(y0)
# print(y1)
        