import glob
import sys

from attr import validate

sys.path.append('core')
import torch.nn.functional as F
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import cv2
from yacs.config import CfgNode as CN
from configs.submission import get_cfg as get_submission_cfg
from configs.tartanair_eval import get_cfg as get_tartanair_cfg
# from configs.kitti_submission import get_cfg as get_kitti_cfg
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
from core.FlowFormer.LatentCostFormer.dimension_test import UNet
import datasets

from core.utils import vars_viz

from core.FlowFormer import build_flowformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import flow_viz

import imageio
import itertools

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from eval import sparsification_plot


def process_image(i, filelist, model, gt_flow, args):

    img1 = np.array(Image.open(filelist[2 * i])).astype(np.uint8)
    img2 = np.array(Image.open(filelist[2 * i + 1])).astype(np.uint8)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1 = img1.cuda()
    img2 = img2.cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    with torch.no_grad():
        flows, vars = model(img1, img2, {})
    if args.flow:
        img = flow_viz.flow_to_image(flows[0].permute(1, 2, 0).cpu().numpy())
        image = Image.fromarray(img)
        image.save(result_path + 'flow/' + str(i).zfill(6) + '.png')
        if not args.error and not args.mse and not args.cov:
            return 0
    flow = np.load(gt_flow[i])
    flow = torch.from_numpy(flow).permute(2, 0, 1).float()
    flow = flow.unsqueeze_(0).cuda()

    mse = torch.pow((flows[0] - flow), 2).squeeze_(0).cpu()
    mse = torch.mean(mse, dim=0)
    if args.cov:
        vars = torch.mean(vars, dim=1).cpu()
        vars.squeeze_(0)
        vars = vars.detach()
        vars = 2 * torch.exp(vars)
        vars = torch.sqrt(vars)
        img = vars_viz.heatmap(vars)
        cv2.imwrite(result_path + 'cov/' + str(i).zfill(6) + '.png', img)
        os.makedirs(result_path + 'cov/' + 'file/', exist_ok=True)
        np.save(result_path + 'cov/' + 'file/' + str(i).zfill(6) + '.npy',
                vars.numpy())
        if not args.error and not args.mse:
            return 0
    if args.mse:
        mse = torch.sqrt(mse)
        img = vars_viz.heatmap(mse)
        cv2.imwrite(result_path + 'mse/' + str(i).zfill(6) + '.png', img)
        os.makedirs(result_path + 'mse/' + 'file/', exist_ok=True)
        np.save(result_path + 'mse/' + 'file/' + str(i).zfill(6) + '.npy',
                mse.numpy())
        return mse.mean()

    if args.error:
        vars = torch.mean(vars, dim=1).cpu().squeeze_(0).detach()
        vars = torch.mean(vars)
        mse = torch.mean(mse)
        print('mse:{}'.format(mse))
        return vars, mse
    return 0


if __name__ == '__main__':
    img_path = 'datasets/eval_data/img/'
    flow_path = 'datasets/eval_data/flow/'
    i = list(range(5001, 120002, 5000))
    model_path = []
    for j in i:
        model_path.append('models/' + str(j) + '_tartanair.pth')

    imglist = glob.glob(img_path + '*.png')
    imglist.sort()
    flowlist = glob.glob(flow_path + '*.npy')
    flowlist.sort()
    length = len(flowlist)
    cfg = get_tartanair_cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow', action='store_true', default=True)
    parser.add_argument('--cov', action='store_true', default=True)
    parser.add_argument('--mse', action='store_true', default=True)
    parser.add_argument('--training_mode', default='cov')
    parser.add_argument('--error', default=False)
    args = parser.parse_args()
    cfg.update(vars(args))
    for modelname in model_path:
        result_path = 'results/eval/' + modelname.split('/')[1].split(
            '_')[0] + '/'
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(result_path + 'flow/', exist_ok=True)
        os.makedirs(result_path + 'cov/', exist_ok=True)
        os.makedirs(result_path + 'mse/', exist_ok=True)
        model = torch.nn.DataParallel(build_flowformer(cfg))
        model.load_state_dict(torch.load(modelname), strict=True)
        model.cuda()
        model.eval()
        results = []
        for i in range(length - 1):
            result = process_image(i, imglist, model, flowlist, args)
            print('{}/{}'.format(i + 1, length))
            if args.error or args.mse:
                results.append(result)

        if args.mse:
            results = np.array(results)
            mean = results.mean(axis=0)
            plt.plot(results, label='mse')
            plt.plot([mean] * length, label='mean')
            plt.savefig(result_path + 'mse.png')
