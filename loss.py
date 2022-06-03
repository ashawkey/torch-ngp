import torch
import torch.nn as nn
import torch.nn.functional as F

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