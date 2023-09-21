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
        _, vars = model(img1, img2, {})

    vars = torch.mean(vars, dim=1).cpu()
    vars.squeeze_(0)
    vars = vars.detach()
    img = vars_viz.flow_var_to_img(vars)
    cv2.imwrite(result_path + 'vars_gru/' + str(i).zfill(6) + '.png', img)
    torch.cuda.empty_cache()

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
    model.load_state_dict(torch.load(cfg.model), strict=True)

    #length = 3
    for i in range(length - 1):
        process_image(i, filelist, model, result_path)
