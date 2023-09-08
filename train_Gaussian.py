from __future__ import print_function, division
import sys

sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import core.datasets as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer, build_gaussian
from core.FlowFormer.LatentCostFormer.dimension_test import UNet
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:

        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    model = nn.DataParallel(build_flowformer(cfg))
    #g_model = nn.DataParallel(build_gaussian(cfg))
    g_model = UNet()
    #loguru_logger.info("Parameter Count: %d" % count_parameters(model))
    loguru_logger.info("MixtureGaussian Parameter Count: %d" %
                       count_parameters(g_model))
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    #freeze the FlowFormer
    for param in model.parameters():
        param.requires_grad = False

    g_model.cuda()
    g_model.train()

    train_loader = datasets.fetch_dataloader(cfg)

    g_optimizer, g_scheduler = fetch_optimizer(g_model, cfg.trainer)
    total_steps = 0

    g_scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(g_model, g_scheduler, cfg)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            g_optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 +
                          stdv * torch.randn(*image1.shape).cuda()).clamp(
                              0.0, 255.0)
                image2 = (image2 +
                          stdv * torch.randn(*image2.shape).cuda()).clamp(
                              0.0, 255.0)

            output = {}
            flow_predictions, mask = model(image1, image2, output)
            # flow_predictions(12)-->mixturegaussian(12)
            # mixturegaussian(12)-->variance(1)-->var_map(1)
            flow_all = torch.cat(flow_predictions, dim=1)

            vars = g_model(flow_all)
            # print(vars.shape)
            # sys.exit()
            loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg,
                                          vars, mask)

            g_scaler.scale(loss).backward()
            g_scaler.unscale_(g_optimizer)
            torch.nn.utils.clip_grad_norm_(g_model.parameters(),
                                           cfg.trainer.clip)
            g_scaler.step(g_optimizer)
            g_scheduler.step()
            g_scaler.update()

            metrics.update(output)
            logger.push(metrics)
            print("Iter: %d, Loss: %.4f" % (total_steps, loss.item()))

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break
            if cfg.autosave_freq and total_steps % cfg.autosave_freq == 0:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1,
                                         cfg.name)
                torch.save(g_model.state_dict(), PATH)
    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}/u_batch=4.pth'
    torch.save(g_model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default='flowformer',
                        help="name your experiment")
    parser.add_argument('--stage',
                        help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'tartanair':
        from configs.tartanair_small import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    #loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
