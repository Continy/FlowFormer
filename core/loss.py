import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily
import sys
import cv2
import numpy as np

MAX_FLOW = 400


def upsample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)


def sequence_loss(flow_preds, flow_gt, valid, cfg, vars, mask):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    method = cfg.mixturegaussian
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
    B, C, H, W = vars.shape
    # k10, k90 = int(B * C * H * W * 0.05), int(B * C * H * W * 0.95)
    # x10, _ = torch.kthvalue(vars.reshape(-1), k10)
    # x90, _ = torch.kthvalue(vars.reshape(-1), k90)
    # vars = torch.clamp(vars, x10, x90)
    if method.method == 'U-net':
        vars_mean = torch.mean(upsample_flow(vars, mask), dim=1)
    elif method.method == 'FlowNetS':
        vars_mean = torch.mean(vars, dim=1)
    else:
        print('wrong method')
        sys.exit()
    if method.training_viz:
        varrr = torch.mean(vars, dim=1)
        varrr = torch.mean(varrr, dim=0)
        varrr = varrr.squeeze_(0).squeeze_(0).detach().cpu().numpy()
        cv2.imwrite('vars.png', varrr * 255)
    mse_loss = torch.mean(mse_loss, dim=1)
    distribution_loss = mse_loss / (2 * torch.exp(2 * vars_mean)) + vars_mean

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
