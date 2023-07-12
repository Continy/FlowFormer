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
import matplotlib.pyplot as plt
from configs.submission import get_cfg as get_submission_cfg
from configs.tartanair_eval import get_cfg as get_tartanair_cfg
# from configs.kitti_submission import get_cfg as get_kitti_cfg
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils

from core.FlowFormer import build_flowformer
from raft import RAFT
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.utils import InputPadder, forward_interpolate
import imageio
import itertools
import glob

TRAIN_SIZE = [432, 960]
KITTI_SIZE = [370, 1226]
TARTANAIR_SIZE = [480, 640]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    hws = [(h, w) for h in hs for w in ws]
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(image_shape), cmap='gray')
    for h, w in hws:
        rect = plt.Rectangle((w, h),
                             patch_size[1],
                             patch_size[0],
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
        ax.add_patch(rect)
    ax.set_axis_off()
    plt.show()
    return hws


import math


def compute_weight(hws,
                   image_shape,
                   patch_size=TRAIN_SIZE,
                   sigma=1.0,
                   wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]),
                          torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h**2 + w**2)**0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw)**2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0],
                                     w:w + patch_size[1]])
    hs = list(range(0, image_shape[0], patch_size[0] - 16))
    ws = list(range(0, image_shape[1], patch_size[1] - 16))
    fig, axs = plt.subplots(nrows=len(hs), ncols=len(ws), figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(weights[0, i, :, :].cpu(), cmap='gray')
        ax.set_axis_off()
    plt.show()

    return patch_weights


def process_image(i, filelist, model, result_path):
    img1 = np.array(Image.open(filelist[i])).astype(np.uint8)
    img2 = np.array(Image.open(filelist[i + 1])).astype(np.uint8)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1 = img1.cuda()
    img2 = img2.cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    with torch.no_grad():
        flow, _ = model(img1, img2)
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    #print(flow)
    #save numpy array to .npy file
    np.save(result_path + str(i).zfill(6) + '.npy', flow)
    flow_img = flow_viz.flow_to_image(flow)
    image = Image.fromarray(flow_img)
    image.save(result_path + str(i).zfill(6) + '.png')
    print('已保存：{}/{}'.format(i + 1, length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='load model')
    parser.add_argument('--eval', help='eval benchmark')
    parser.add_argument('--small', action='store_true', help='use small model')
    args = parser.parse_args()
    cfg = get_tartanair_cfg()
    cfg.latentcostformer.decoder_depth = 32
    #print(cfg)
    result_path = 'results/' + args.eval + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    img_path = 'datasets/KITTI/image_left/'
    img_path = 'datasets/abandonedfactory/Easy/P001/image_left/'
    pattern = os.path.join(img_path, '*_left.png')
    filelist = glob.glob(pattern)
    length = len(filelist)
    print(length)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()
    #length = 3
    for i in range(length - 1):
        process_image(i, filelist, model, result_path)
