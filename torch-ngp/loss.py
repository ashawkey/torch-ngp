import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def mape_loss(pred, target, reduction='mean'):
    # pred, target: [B, 1], torch tenspr
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    if reduction == 'mean':
        loss = loss.mean()
    
    return loss

def huber_loss(pred, target, delta=0.1, reduction='mean'):
    rel = (pred - target).abs()
    sqr = 0.5 / delta * rel * rel
    loss = torch.where(rel > delta, rel - 0.5 * delta, sqr)

    if reduction == 'mean':
        loss = loss.mean()

    return loss


# ref: https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
class EffDistLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, m, interval):
        '''
        Efficient O(N) realization of distortion loss.
        There are B rays each with N sampled points.
        w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
        m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
        interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
        '''
        n_rays = np.prod(w.shape[:-1])
        wm = (w * m)
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)

        w_total = w_cumsum[..., [-1]]
        wm_total = wm_cumsum[..., [-1]]
        w_prefix = torch.cat([torch.zeros_like(w_total), w_cumsum[..., :-1]], dim=-1)
        wm_prefix = torch.cat([torch.zeros_like(wm_total), wm_cumsum[..., :-1]], dim=-1)
        loss_uni = (1/3) * interval * w.pow(2)
        loss_bi = 2 * w * (m * w_prefix - wm_prefix)
        if torch.is_tensor(interval):
            ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval)
            ctx.interval = None
        else:
            ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total)
            ctx.interval = interval
        ctx.n_rays = n_rays
        return (loss_bi.sum() + loss_uni.sum()) / n_rays

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        interval = ctx.interval
        n_rays = ctx.n_rays
        if interval is None:
            w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval = ctx.saved_tensors
        else:
            w, m, wm, w_prefix, w_total, wm_prefix, wm_total = ctx.saved_tensors
        grad_uni = (1/3) * interval * 2 * w
        w_suffix = w_total - (w_prefix + w)
        wm_suffix = wm_total - (wm_prefix + wm)
        grad_bi = 2 * (m * (w_prefix - w_suffix) + (wm_suffix - wm_prefix))
        grad = grad_back * (grad_bi + grad_uni) / n_rays
        return grad, None, None, None

eff_distloss = EffDistLoss.apply
