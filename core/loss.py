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
    n_predictions = len(flow_preds)
    flow_gt_thresholds = [5, 10, 20]

    mse_loss = torch.zeros_like(flow_gt)
    vars_mean = torch.zeros_like(flow_gt)
    distribution_loss = torch.zeros_like(flow_gt)

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    if cfg.training_viz:
        viz = torch.mean(
            vars_mean,
            dim=0).squeeze_(0).squeeze_(0).detach().cpu().numpy() * 255
        cv2.imwrite('vars.png', viz)
    if cfg.training_mode == 'flow':
        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            mse_loss += i_weight * (valid[:, None] * i_loss)

            epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid.view(-1)]
            metrics = {
                'loss': mse_loss.mean().item(),
                'epe': epe.mean().item(),
                '1px': (epe < 1).float().mean().item(),
                '3px': (epe < 3).float().mean().item(),
                '5px': (epe < 5).float().mean().item(),
            }

        return mse_loss.mean(), metrics
    if cfg.training_mode == 'cov':
        i_loss = (flow_preds - flow_gt)**2
        mse_loss += (valid[:, None] * i_loss)
        vars_mean = torch.mean(vars, dim=1)
        mse_loss = torch.mean(mse_loss, dim=1)
        distribution_loss = mse_loss / (2 *
                                        torch.exp(2 * vars_mean)) + vars_mean
        metrics = {
            'vars': vars_mean.float().mean().item(),
            'distribution_loss': distribution_loss.float().mean().item(),
            'mse_loss': mse_loss.float().mean().item()
        }
        return distribution_loss.mean(), metrics
    else:
        print('training mode not supported')
        sys.exit()
