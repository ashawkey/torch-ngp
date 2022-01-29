import torch

from sdf.netowrk import SDFNetwork
from sdf.netowrk_ff import SDFNetwork as SDFNetwork_FF
from sdf.provider import SDFDataset
from sdf.utils import *

from loss import mape_loss

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")

    opt = parser.parse_args()

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        Network = SDFNetwork_FF
    else:
        Network = SDFNetwork
    seed_everything(opt.seed)

    train_dataset = SDFDataset(opt.path, size=100, num_samples=2**18)
    valid_dataset = SDFDataset(opt.path, size=1, num_samples=2**18) # just a dummy

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    model = Network(encoding="hashgrid")
    #model = SDFNetwork(encoding="frequency", num_layers=8, skips=[4], hidden_dim=256)

    print(model)

    criterion = mape_loss # torch.nn.L1Loss()

    #optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-6, eps=1e-15)
    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': model.encoder.parameters()},
        {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
    ], lr=1e-4, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer('ngp', model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1)

    trainer.train(train_loader, valid_loader, 20)

    # save a high resolution mesh
    trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)
