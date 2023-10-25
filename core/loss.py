import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily
import sys
import cv2
import numpy as np


def sequence_loss(flow_preds, flow_gt, valid, cfg, vars):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    method = cfg.mixturegaussian
    n_predictions = len(flow_preds)
    mse_loss = torch.zeros_like(flow_gt)
    vars_mean = torch.zeros_like(flow_gt)
    distribution_loss = torch.zeros_like(flow_gt)

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    i_loss = (flow_preds - flow_gt).abs()
    mse_loss += (valid[:, None] * i_loss)

    vars_mean = torch.mean(vars, dim=1)
    mse_loss = torch.mean(mse_loss, dim=1)

    if method.training_viz:
        viz = torch.mean(
            vars_mean,
            dim=0).squeeze_(0).squeeze_(0).detach().cpu().numpy() * 255
        cv2.imwrite('vars.png', viz)
    vars_mean = vars_mean.clamp(min=1e-6, max=20)
    #distribution_loss = mse_loss / (2 * torch.exp(2 * vars_mean)) + vars_mean
    distribution_loss = (mse_loss / vars_mean - 1)**2
    #cut off pixels with large variance
    distribution_loss = distribution_loss.clamp(max=1e2)

    metrics = {
        'vars': vars_mean.float().mean().item(),
        'distribution_loss': distribution_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }

    return distribution_loss.mean(), metrics
