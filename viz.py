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

TRAIN_SIZE = [432, 960]
KITTI_SIZE = [370, 1226]
TARTANAIR_SIZE = [480, 640]


def process_image(i, filelist, model, gt_flow):
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

    # flow = flows[0].permute(1, 2, 0).cpu().numpy()
    # np.save(result_path + str(i).zfill(6) + '.npy', flow)
    # flow_img = flow_viz.flow_to_image(vars)
    #vars = upsample_flow(vars, mask)
    mse = (flows[0] - flow).abs().squeeze_(0).cpu()
    mse = torch.mean(mse, dim=0)
    vars_mean = vars.mean().cpu()
    mse_mean = mse.mean().cpu()
    #img = vars_viz.flow_var_to_img(mse)

    # cv2.imwrite(result_path + 'mse/' + str(i).zfill(6) + '.png', img)
    torch.cuda.empty_cache()
    # image = Image.fromarray(flow_img)
    # image.save(result_path + str(i).zfill(6) + '.png')
    print('Saved：{}/{}'.format(i + 1, length))
    return [vars_mean, mse_mean]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='load model')
    parser.add_argument('--eval', help='eval benchmark')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--datadir',
                        help='dataset dir',
                        default='abandonedfactory/Easy/P001/')
    args = parser.parse_args()
    cfg = get_tartanair_cfg()

    #print(cfg)
    result_path = 'results/' + args.eval + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    gt_flow = 'datasets/' + args.datadir + 'flow/'
    img_path = 'datasets/' + args.datadir + 'image_left/'
    pattern1 = os.path.join(img_path, '*.png')
    pattern2 = os.path.join(gt_flow, '*.npy')
    filelist1 = glob.glob(pattern1)
    filelist2 = glob.glob(pattern2)
    length = len(filelist1)
    print(length)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model), strict=True)

    model.cuda()

    model.eval()

    #length = 3
    results = []
    for i in range(length - 1):
        result = process_image(i, filelist1, model, filelist2)
        results.append(result)
    results = np.array(results)
    import matplotlib.pyplot as plt
    x = results[:, 0]
    y = results[:, 1]
    name = 'gru'
    np.save(name + '.npy', results)
    plt.plot(x, y, 'o')
    corr_coef = np.corrcoef(x, y)[0, 1]
    plt.xlabel('mean of variance')
    plt.ylabel('mean of mse')
    plt.title('correlation coefficient:{}'.format(corr_coef))
    plt.savefig(name + '.png')
    plt.show()
