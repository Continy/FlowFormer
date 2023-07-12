from matplotlib import pyplot as plt
import os
import glob
import argparse
import torch


def cal_EPE(gt_flow, est_flow):
    return torch.norm(gt_flow - est_flow, 2, 1).mean()


origin_path = 'C:/Users/zihaozhang/Desktop/git/FlowFormer-Official/datasets/abandonedfactory/Easy/P001/image_left/'
gt_path = 'C:/Users/zihaozhang/Desktop/git/FlowFormer-Official/datasets/abandonedfactory/Easy/P001/flow_img/'
flowformer_path = 'results/tartanair/things/'
pwcnet_path = 'results/tartanair/PWCNet/'
save_path = 'compare/tartanair/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
pattern = os.path.join(flowformer_path, '*.png')
filelist = sorted(glob.glob(pattern))
for i in range(367, len(filelist)):
    #4x4
    plt.subplot(221)
    plt.imshow(plt.imread(origin_path + str(i).zfill(6) + '_left.png'))
    plt.axis('off')
    plt.title('Source Image')
    plt.subplot(222)
    plt.imshow(plt.imread(gt_path + str(i).zfill(6) + '.png'))
    plt.axis('off')
    plt.title('Ground Truth')
    plt.savefig(save_path + str(i).zfill(6) + '.png')
    plt.subplot(223)
    plt.imshow(plt.imread(flowformer_path + str(i).zfill(6) + '.png'))
    plt.axis('off')
    plt.title('FlowFormer')
    plt.subplot(224)
    plt.imshow(plt.imread(pwcnet_path + str(i).zfill(6) + '.png'))
    plt.axis('off')
    plt.title('PWCNet')
    plt.savefig(save_path + str(i).zfill(6) + '.png')
    plt.close()
    print('已保存：{}/{}'.format(i + 1, len(filelist)))
