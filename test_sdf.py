from sdf.netowrk import SDFNetwork
from sdf.utils import *

import argparse

CLIP_SDF = None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    model = SDFNetwork(encoding="hashgrid", clip_sdf=CLIP_SDF)
    #model = SDFNetwork(encoding="frequency", num_layers=8, skips=[4], hidden_dim=256, clip_sdf=CLIP_SDF)

    print(model)

    trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)

    trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

