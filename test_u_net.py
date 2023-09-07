import torch
import torch.nn as nn
#from core.FlowFormer.LatentCostFormer.swin_transformer_unet_skip_expand_decoder_sys import BasicLayer, PatchEmbed, PatchExpand, FinalPatchExpand_X4, BasicLayer_up, PatchMerging
from core.FlowFormer.LatentCostFormer.dimension_test import UNet

x = torch.randn(1, 2, 480, 640)
model = UNet()
out = model(x)
print(out.shape)