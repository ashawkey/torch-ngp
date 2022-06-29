import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from tensoRF.utils import *

from scipy.spatial.transform import Rotation as Rot

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--compose', action='store_true', help="compose mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr0', type=float, default=2e-2, help="initial learning rate for embeddings")
    parser.add_argument('--lr1', type=float, default=1e-3, help="initial learning rate for networks")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--l1_reg_weight', type=float, default=1e-5)

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--resolution0', type=int, default=128)
    parser.add_argument('--resolution1', type=int, default=300)
    parser.add_argument("--upsample_model_steps", type=int, action="append", default=[2000, 3000, 4000, 5500, 7000])

    ### dataset options
    parser.add_argument('--color_space', type=str, default='linear', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True    

    print(opt)
    seed_everything(opt.seed)

    assert opt.cuda_ray, 'CCNeRF only supports CUDA raymarching mode for now.'

    from tensoRF.network_cc import NeRFNetwork as CCNeRF

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # compose mode
    if opt.compose:

        # init an empty scene. (necessary!)
        model = CCNeRF(
            rank_vec_density=[1],
            rank_mat_density=[1],
            rank_vec=[1],
            rank_mat=[1],
            resolution=[1] * 3, # fake resolution
            bound=opt.bound, # a large bound is needed
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        ).to(device)
        
        # helper function to load a single model
        def load_model(path):
            checkpoint_dict = torch.load(path, map_location=device)
            model = CCNeRF(
                rank_vec_density=checkpoint_dict['rank_vec_density'],
                rank_mat_density=checkpoint_dict['rank_mat_density'],
                rank_vec=checkpoint_dict['rank_vec'],
                rank_mat=checkpoint_dict['rank_mat'],
                resolution=checkpoint_dict['resolution'],
                bound=opt.bound,
                cuda_ray=opt.cuda_ray,
                density_scale=1,
                min_near=opt.min_near,
                density_thresh=opt.density_thresh,
                bg_radius=opt.bg_radius,
            ).to(device)

            model.load_state_dict(checkpoint_dict['model'], strict=False)
            return model

        # compose example
        hotdog = load_model('trial_cc_hotdog/checkpoints/64_16-64_64.pth')
        chair = load_model('trial_cc_chair/checkpoints/64_16-64_64.pth')
        ficus = load_model('trial_cc_ficus/checkpoints/64_16-64_64.pth')

        model.compose(hotdog, s=0.4, t=np.array([0, 0.2, 0]))
        model.compose(ficus, s=0.6, t=np.array([0, 0, -0.5]), R=Rot.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix())
        model.compose(chair, s=0.6, t=np.array([0, 0, 0.5]), R=Rot.from_euler('zyx', [0, -90, 0], degrees=True).as_matrix())
        model.compose(chair, s=0.6, t=np.array([-0.5, 0, 0]), R=Rot.from_euler('zyx', [0, 180, 0], degrees=True).as_matrix())
        model.compose(chair, s=0.6, t=np.array([0.5, 0, 0]), R=Rot.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix())

        # tell trainer not to load ckpt again
        opt.ckpt = 'scratch'

        
    # single model mode
    else:
        model = CCNeRF(
            resolution=[opt.resolution0] * 3,
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        ).to(device)
        
    print(model)

    if opt.test:

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            # compose mode have no gt, do not evaulate
            if opt.compose:
                trainer.test(test_loader, save_path=os.path.join(opt.workspace, 'compose'))
            elif test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.

            #trainer.save_mesh(resolution=256, threshold=0.1)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr0, opt.lr1), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=50)

        # calc upsample target resolutions
        upsample_resolutions = (np.round(np.exp(np.linspace(np.log(opt.resolution0), np.log(opt.resolution1), len(opt.upsample_model_steps) + 1)))).astype(np.int32).tolist()[1:]
        print('upsample_resolutions:', upsample_resolutions)
        trainer.upsample_resolutions = upsample_resolutions

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            # save and test at multiple compression levels
            K = model.K[0]
            rank_vec_density = model.rank_vec_density[0][::-1]
            rank_mat_density = model.rank_mat_density[0][::-1]
            rank_vec = model.rank_vec[0][::-1]
            rank_mat = model.rank_mat[0][::-1]

            model.finalize()
            print(f'[INFO] ===== finalized model =====')
            print(model)

            for k in range(K):
                model.compress((rank_vec_density[k], rank_mat_density[k], rank_vec[k], rank_mat[k]))
                name = f'{rank_vec_density[k]}_{rank_mat_density[k]}-{rank_vec[k]}_{rank_mat[k]}'
                print(f'[INFO] ===== compressed at {name} =====')
                print(model)
                trainer.save_checkpoint(name, full=False, remove_old=False)

                if test_loader.has_gt:
                    trainer.evaluate(test_loader, name=name) # blender has gt, so evaluate it.
                else:
                    trainer.test(test_loader, name=name) # colmap doesn't have gt, so just test.

