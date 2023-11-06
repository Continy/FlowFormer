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
from core.FlowFormer import build_flowformer

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


from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]
#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    model = nn.DataParallel(build_flowformer(cfg))

    #loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=False)

    model.cuda()
    model.train()
    if cfg.training_mode == 'flow':
        #freeze the Covariance Decoder
        for param in model.module.memory_decoder.gaussian.parameters():
            param.requires_grad = False
        optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    if cfg.training_mode == 'cov':
        #freeze the FlowFormer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.module.memory_decoder.gaussian.parameters():
            param.requires_grad = True
        optimizer, scheduler = fetch_optimizer(
            model.module.memory_decoder.gaussian, cfg.trainer)
    train_loader = datasets.fetch_dataloader(cfg)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    if cfg.log:
        logger = Logger(model, scheduler, cfg)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
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
            flow_predictions, covs = model(image1, image2, output)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg,
                                          covs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.trainer.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            if cfg.log:
                metrics.update(output)
                logger.push(metrics)
            if total_steps % 5 == 0:
                print("Iter: %d, Loss: %.4f" % (total_steps, loss.item()))

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break
            if cfg.autosave_freq and total_steps % cfg.autosave_freq == 0 and cfg.log:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1,
                                         cfg.name)
                torch.save(model.state_dict(), PATH)
    if cfg.log:
        logger.close()
        PATH = cfg.log_dir + '/final'
        torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/big_full.pth'
    torch.save(model.state_dict(), PATH)

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
    parser.add_argument('--training_mode',
                        default='cov',
                        help='flow or covariance')
    parser.add_argument('--log', action='store_true', help='disable logging')
    parser.add_argument('--big', action='store_true', help='use big model')

    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'tartanair' and not args.big:
        from configs.tartanair_small import get_cfg
    elif args.stage == 'tartanair' and args.big:
        from configs.tartanair_big import get_cfg
    cfg = get_cfg()
    cfg.update(vars(args))
    if args.log:
        process_cfg(cfg)
        loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
        cfg.log = True
    #loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
