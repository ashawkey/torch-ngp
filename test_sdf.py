from sdf.netowrk import SDFNetwork
from sdf.netowrk_ff import SDFNetwork as SDFNetwork_FF
from sdf.utils import *

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

    model = Network(encoding="hashgrid")
    #model = SDFNetwork(encoding="frequency", num_layers=8, skips=[4], hidden_dim=256, clip_sdf=CLIP_SDF)

    print(model)

    trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)

    trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

