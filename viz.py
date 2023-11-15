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
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import flow_viz

import imageio
import itertools
import glob
import matplotlib.pyplot as plt

TRAIN_SIZE = [432, 960]
KITTI_SIZE = [370, 1226]
TARTANAIR_SIZE = [480, 640]


def error(mse, vars):
    return np.mean((mse - vars)**2)


def process_image(i, filelist, model, gt_flow, args):
    img1 = np.array(Image.open(filelist[i])).astype(np.uint8)
    img2 = np.array(Image.open(filelist[i + 1])).astype(np.uint8)

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
        print('{}/{}'.format(i + 1, length))

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
        print('{}/{}'.format(i + 1, length))
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
        print('{}/{}'.format(i + 1, length))
        return mse.mean()

    if args.error:
        vars = torch.mean(vars, dim=1).cpu().squeeze_(0).detach()
        vars = torch.mean(vars)
        mse = torch.mean(mse)
        print('{}/{}'.format(i + 1, length))
        print('mse:{}'.format(mse))
        return vars, mse
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--small',
                        action='store_true',
                        help='Using small model')
    parser.add_argument('--gtflowdir',
                        help='Ground truth flow dir',
                        default='datasets/eval_data/flow/')
    parser.add_argument('--imgdir',
                        help='image dir',
                        default='datasets/eval_data/img/')
    parser.add_argument('--savepath',
                        help='store the results',
                        default='results/a/')
    parser.add_argument('--flow', help='visualize flow', action='store_true')
    parser.add_argument('--error', help='visualize error', action='store_true')
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
    result_path = args.savepath
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path + 'flow/', exist_ok=True)
    os.makedirs(result_path + 'cov/', exist_ok=True)
    os.makedirs(result_path + 'mse/', exist_ok=True)
    img_path = args.imgdir
    gt_flow = args.gtflowdir
    pattern1 = os.path.join(img_path, '*.png')
    pattern2 = os.path.join(gt_flow, '*.npy')
    filelist1 = sorted(glob.glob(pattern1))
    filelist2 = sorted(glob.glob(pattern2))
    length = len(filelist1)
    print('Find {} pairs'.format(length))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model), strict=True)
    model.cuda()
    model.eval()
    results = []
    length = 2
    for i in range(length - 1):
        result = process_image(i, filelist1, model, filelist2, args)
        if args.error or args.mse:
            results.append(result)

    if args.mse:
        results = np.array(results)
        mean = results.mean(axis=0)
        plt.plot(results, label='mse')
        plt.plot([mean] * length, label='mean')
        plt.legend()
        plt.savefig(result_path + 'mse.png')
    if args.error:
        results = np.array(results)
        vars = results[:, 0]
        mse = results[:, 1]
        # calculate the variance of mse
        mean = mse.mean()
        simple_variance = np.mean((mse - mean)**2)
        # turn shape of simple_variance to shape of mse
        simple_variance = np.array([simple_variance] * (length - 1))
        simple_error = []
        vars_error = []

        # for i in range(length-1):
        #     # simple_error.append(error(mse[:i+1],simple_variance[:i+1]))
        #     # vars_error.append(error(mse[:i+1],vars[:i+1]))
        #     simple_error.append(vars[i])
        #     vars_error.append(np.log(mse[i]))
        # np.save(result_path+'vars.npy',simple_error)
        # plt.plot(simple_error,label='simple')
        # plt.plot(vars_error,label='vars')
        # plt.plot([mean]*length,label='mean')
        # plt.plot([np.log(mean)]*length,label='log_mean')
        # plt.legend()
        # plt.savefig(result_path + 'error.png')
