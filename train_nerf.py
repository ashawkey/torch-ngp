import torch

from nerf.provider import NeRFDataset
from nerf.utils import *

import argparse

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    # the default setting for fox.
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

    opt = parser.parse_args()

    print(opt)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    seed_everything(opt.seed)

    train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale)
    valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
    
    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
    )

    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)

    print(model)

    criterion = torch.nn.SmoothL1Loss()

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)

    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=10)
    
    trainer.train(train_loader, valid_loader, 200)

    # test dataset
    #trainer.save_mesh()
    test_dataset = NeRFDataset(opt.path, 'test', radius=opt.radius, n_test=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
