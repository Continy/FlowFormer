import torch
import torch.nn as nn
import sys

sys.path.append('core')
#from core.FlowFormer.LatentCostFormer.swin_transformer_unet_skip_expand_decoder_sys import BasicLayer, PatchEmbed, PatchExpand, FinalPatchExpand_X4, BasicLayer_up, PatchMerging
from core.FlowFormer.LatentCostFormer.dimension_test import UNet
import cv2

img = cv2.imread(
    'D:/gits/FlowFormer/results/tartanair/small/things/000000.png')

#转换为tensor
img = torch.from_numpy(img)
#转换为[1, 3, 480, 640]
img = img.permute(2, 0, 1).unsqueeze(0)
#转换为[1, 24, 480, 640]
img = img.repeat(1, 8, 1, 1)
img = img.float()
img = img / 255.0
img = torch.ones(1, 24, 480, 640)
#zero = torch.zeros(1, 24, 480, 320)
#img = torch.cat((img, zero), dim=3)
model = UNet()
# model.load_state_dict(torch.load('checkpoints/tartanair/u_batch=4.pth'))
out = model(img)
print(out.shape)
out = torch.mean(out, dim=1)

cv2.imwrite('test2.png', out[0].detach().numpy() * 255)
