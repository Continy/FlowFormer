import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily
import sys

MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, cfg, vars):
    """ Loss function defined over sequence of flow predictions """
    B, C, H, W = flow_gt.shape
    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)
    mse_loss = torch.zeros_like(flow_gt)
    vars_mean = torch.zeros_like(flow_gt)
    distribution_loss = torch.zeros_like(flow_gt)
    #mse_loss, vars_mean, distribution_loss = 0.0, 0.0, 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        mse_loss += i_weight * (valid[:, None] * i_loss)
    vars_mean = torch.mean(vars, dim=1)
    distribution_loss = mse_loss / (2 * torch.exp(2 * vars_mean)) + vars_mean
    #print(distribution_loss.mean())
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'vars': vars_mean.float().mean().item(),
        'distribution_loss': distribution_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    return distribution_loss.mean(), metrics
