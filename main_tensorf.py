import torch
import argparse

from tensorf.provider import NeRFDataset
from tensorf.utils import *

#torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    ### training options
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    ### dataset options
    parser.add_argument('--mode', type=str, default='blender', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, fasten training but use more GPU memory")
    parser.add_argument('--bound', type=float, default=1.5, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=1.0, help="scale camera location into box(-bound, bound)")
    ### tensorf options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument("--lr_init", type=float, default=2e-2, help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    parser.add_argument('--N_voxel_init', type=int, default=128**3)
    parser.add_argument('--N_voxel_final', type=int, default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append", default=[2000,3000,4000,5500,7000])
    parser.add_argument("--update_AlphaMask_list", type=int, action="append", default=[]) # [2000,4000]
    parser.add_argument('--lindisp', default=False, action="store_true", help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='relu')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6, help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)
    parser.add_argument("--L1_weight_inital", type=float, default=8e-5, help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=4e-5, help='loss weight')
    #parser.add_argument("--Ortho_weight", type=float, default=0.0, help='loss weight')
    #parser.add_argument("--TV_weight_density", type=float, default=0.0, help='loss weight')
    #parser.add_argument("--TV_weight_app", type=float, default=0.0, help='loss weight')
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append", default=[16, 16, 16])
    parser.add_argument("--n_lamb_sh", type=int, action="append", default=[48, 48, 48])
    parser.add_argument("--data_dim_color", type=int, default=27)
    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001, help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.08, help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25, help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10, help='shift density in softplus; making density = 0  when feature == 0')
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_Fea", help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6, help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=2, help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=2, help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128, help='hidden feature channel in MLP')

    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)

    from tensorf.network import TensorVMSplit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aabb = (torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]) * opt.bound).to(device)
    reso_cur = N_to_reso(opt.N_voxel_init, aabb)
    nSamples = 512 # min(opt.nSamples, cal_n_samples(reso_cur, opt.step_ratio))
    near_far = [2.0, 6.0] # fixed for blender
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(opt.N_voxel_init), np.log(opt.N_voxel_final), len(opt.upsamp_list)+1))).long()).tolist()[1:]

    model = TensorVMSplit(
        aabb, reso_cur, device,
        density_n_comp=opt.n_lamb_sigma, appearance_n_comp=opt.n_lamb_sh, 
        app_dim=opt.data_dim_color, near_far=near_far,
        shadingMode=opt.shadingMode, alphaMask_thres=opt.alpha_mask_thre, density_shift=opt.density_shift, distance_scale=opt.distance_scale,
        pos_pe=opt.pos_pe, view_pe=opt.view_pe, fea_pe=opt.fea_pe, 
        featureC=opt.featureC, step_ratio=opt.step_ratio, fea2denseAct=opt.fea2denseAct,
        cuda_ray=opt.cuda_ray,
    )
    
    print(model)

    criterion = torch.nn.MSELoss() # HuberLoss(delta=0.1)

    ### test mode
    if opt.test:

        trainer = Trainer('tensorf', vars(opt), model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint='latest')

        test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True)

        if opt.mode == 'blender':
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        else:
            trainer.test(test_loader) # colmap doesn't have gt, so just test.
    
    else:

        
        optimizer = lambda model: torch.optim.Adam(model.get_optparam_groups(opt.lr_init, opt.lr_basis), betas=(0.9, 0.99))

        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.33)

        trainer = Trainer('tensorf', vars(opt), model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, metrics=[PSNRMeter()], use_checkpoint='scratch', eval_interval=50)

        # attach extra things
        trainer.aabb = aabb
        trainer.reso_cur = reso_cur
        trainer.nSamples = nSamples
        trainer.near_far = near_far
        trainer.L1_reg_weight = opt.L1_weight_inital
        trainer.N_voxel_list = N_voxel_list

        train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale, preload=opt.preload)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, pin_memory=True)

        trainer.train(train_loader, valid_loader, 300)

        # also test
        test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True)
        
        if opt.mode == 'blender':
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        else:
            trainer.test(test_loader) # colmap doesn't have gt, so just test.