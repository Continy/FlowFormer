import sys

from attr import validate

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
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


def process_image(i, filelist, model, g_model, result_path):
    img1 = np.array(Image.open(filelist[i])).astype(np.uint8)
    img2 = np.array(Image.open(filelist[i + 1])).astype(np.uint8)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1 = img1.cuda()
    img2 = img2.cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    with torch.no_grad():
        flows, mask = model(img1, img2, {})
        flow_all = torch.cat(flows, dim=1)
        vars = g_model(flow_all)
    # flow = flows[0].permute(1, 2, 0).cpu().numpy()
    # np.save(result_path + str(i).zfill(6) + '.npy', flow)
    # flow_img = flow_viz.flow_to_image(vars)
    #vars = upsample_flow(vars, mask)
    B, C, H, W = vars.shape
    # k50, k95 = int(B * C * H * W * 0.95), int(B * C * H * W * 1)
    # x50, _ = torch.kthvalue(vars.reshape(-1), k50)
    # x95, _ = torch.kthvalue(vars.reshape(-1), k95)
    # vars = torch.clamp(vars, x50, x95)
    vars = torch.mean(vars, dim=1).cpu()
    vars.squeeze_(0)
    # img = vars_viz.flow_var_to_img(vars)
    vars = vars.detach().numpy()
    cv2.imwrite(result_path + 'vars_flownets_10/' + str(i).zfill(6) + '.png',
                vars * 255)
    torch.cuda.empty_cache()
    # image = Image.fromarray(flow_img)
    # image.save(result_path + str(i).zfill(6) + '.png')
    print('Saved：{}/{}'.format(i + 1, length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='load model')
    parser.add_argument('--eval', help='eval benchmark')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--datadir',
                        help='dataset dir',
                        default='abandonedfactory/Easy/P001/image_left/')
    args = parser.parse_args()
    cfg = get_tartanair_cfg()
    method = cfg.mixturegaussian
    #print(cfg)
    result_path = 'results/' + args.eval + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    img_path = 'datasets/' + args.datadir

    pattern = os.path.join(img_path, '*.png')
    filelist = glob.glob(pattern)
    length = len(filelist)
    print(length)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    if method.method == 'U-net':
        g_model = UNet()
    elif method.method == 'FlowNetS':
        g_model = torch.nn.DataParallel(build_gaussian(cfg))
    else:
        print('wrong method')
        sys.exit()
    g_model.load_state_dict(torch.load(cfg.g_model))
    model.cuda()
    g_model.cuda()
    model.eval()
    g_model.eval()
    #length = 3
    for i in range(length - 1):
        process_image(i, filelist, model, g_model, result_path)
