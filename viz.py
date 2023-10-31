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
from core.FlowFormer import build_gaussian
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
import itertools
import glob
import matplotlib.pyplot as plt

TRAIN_SIZE = [432, 960]
KITTI_SIZE = [370, 1226]
TARTANAIR_SIZE = [480, 640]


def process_image(i, filelist, model, gt_flow, args):
    img1 = np.array(Image.open(filelist[i])).astype(np.uint8)
    img2 = np.array(Image.open(filelist[i + 1])).astype(np.uint8)
    flow = np.load(gt_flow[i])
    flow = torch.from_numpy(flow).permute(2, 0, 1).float()
    flow = flow.unsqueeze_(0).cuda()

    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1 = img1.cuda()
    img2 = img2.cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    with torch.no_grad():
        flows, vars = model(img1, img2, {})

    mse = (flows[0] - flow).abs().squeeze_(0).cpu()
    mse = torch.mean(mse, dim=0)
    if args.mse:
        img = vars_viz.heatmap(mse)
        cv2.imwrite(result_path + str(i).zfill(6) + '.png', img)
    if args.cov:
        vars = torch.mean(vars, dim=1).cpu()
        vars.squeeze_(0)
        vars = vars.detach()
        vars = torch.sqrt(vars)
        img = vars_viz.heatmap(vars)
        cv2.imwrite(result_path + str(i).zfill(6) + '.png', img)
    if args.error:

        vars = torch.mean(vars, dim=1).cpu()
        vars.squeeze_(0)
        vars = vars.detach()
        vars = torch.sqrt(vars)
        error = (mse / vars - 1)**2
        img = vars_viz.colorbar(error)
        plt.colorbar(img)
        plt.savefig(result_path + str(i).zfill(6) + '.png')
        plt.clf()
        print('{}/{}'.format(i + 1, length))
        return error.mean()
    vars_mean = vars.mean().cpu()
    mse_mean = mse.mean().cpu()

    torch.cuda.empty_cache()
    print('{}/{}'.format(i + 1, length))
    return [vars_mean, mse_mean]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--small',
                        action='store_true',
                        help='Using small model')
    parser.add_argument('--gtflowdir',
                        help='Ground truth flow dir',
                        default='datasets/abandonedfactory/Easy/P000/flow/')
    parser.add_argument(
        '--imgdir',
        help='image dir',
        default='datasets/abandonedfactory/Easy/P000/image_left/')
    parser.add_argument('--savepath',
                        help='store the results',
                        default='results/tartanair/small/things/P000_error/')
    parser.add_argument('--error',
                        help='visualize error, (mse/vars-1)^2',
                        action='store_true')
    parser.add_argument('--cov',
                        help='visualize covariance',
                        action='store_true')
    parser.add_argument('--mse', help='visualize mse', action='store_true')
    parser.add_argument('--training_mode',
                        default='cov',
                        help='flow or covariance')
    args = parser.parse_args()
    cfg = get_tartanair_cfg()
    cfg.update(vars(args))
    #print(cfg)
    result_path = args.savepath

    img_path = args.imgdir
    gt_flow = args.gtflowdir
    pattern1 = os.path.join(img_path, '*.png')
    pattern2 = os.path.join(gt_flow, '*.npy')
    filelist1 = glob.glob(pattern1)
    filelist2 = glob.glob(pattern2)
    length = len(filelist1)
    print('Find {} pairs'.format(length))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model), strict=True)
    model.cuda()
    model.eval()

    results = []
    for i in range(length - 1):
        result = process_image(i, filelist1, model, filelist2, args)
        if args.error:
            results.append(result)
    if args.error:
        results = np.array(results)
        mean = results.mean(axis=0)
        plt.plot(results, label='vars')
        plt.plot([mean] * length, label='mean')
        plt.legend()
        plt.savefig(result_path + 'error.png')
