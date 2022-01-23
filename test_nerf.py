import torch

from nerf.network import NeRFNetwork
from nerf.provider import NeRFDataset
from nerf.utils import *

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096) # lower if OOM
    
    parser.add_argument('--radius', type=float, default=2, help="assume the camera is located on sphere(0, radius))")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in sphere(0, size)")

    opt = parser.parse_args()

    print(opt)

    seed_everything(opt.seed)

    model = NeRFNetwork(encoding="hashgrid", encoding_dir="sphere_harmonics", num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64)

    print(model)
    
    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, use_checkpoint='latest')

    # test dataset
    test_dataset = NeRFDataset(opt.path, 'test', downscale=1, radius=opt.radius)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
