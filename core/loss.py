import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily
import sys

MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, cfg, gaussian):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    #将means&vars&weights变成(1,60,80,5,2)
    means, vars, weights = gaussian
    means = means.permute(0, 3, 4, 1, 2)
    vars = vars.permute(0, 3, 4, 1, 2)
    weights = weights.permute(0, 2, 3, 1)
    vars = torch.diag_embed(vars)
    dist = MultivariateNormal(means, vars)
    dist = MixtureSameFamily(torch.distributions.Categorical(probs=weights),
                             dist)

    lable = torch.linspace(-1, 1, n_predictions).to(weights.device)
    i_weights = torch.zeros(n_predictions, cfg.batch_size, 480,
                            640).to(weights.device)
    for i in range(n_predictions):
        #将lable[i]变成(1,60,80,1)
        target = lable[i].expand_as(torch.zeros(1, 60, 80, 1))
        i_weight = dist.log_prob(target)
        i_weight = torch.exp(i_weight)
        i_weight = i_weight.unsqueeze(0)
        i_weight = nn.Upsample(scale_factor=8,
                               mode='bilinear',
                               align_corners=True)(i_weight)
        i_weight = i_weight.squeeze(0)
        i_weights[i] = i_weight

    i_weights += 1e-6
    i_weights = i_weights / i_weights.sum(dim=0, keepdim=True)
    for i in range(n_predictions):
        i_loss = (flow_preds[i] - flow_gt).abs()

        w = i_weights[i].unsqueeze(1)

        flow_loss += (w * (valid[:, None] * i_loss))
    flow_loss = flow_loss.mean()

    var_loss = flow_loss / (2 * torch.exp(2 * vars.mean())) + vars.mean()

    # # old version
    # for i in range(n_predictions):
    #     i_weight = gamma**(n_predictions - i - 1)
    #     i_loss = (flow_preds[i] - flow_gt).abs()
    #     flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'vars': vars.mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    return var_loss, metrics
