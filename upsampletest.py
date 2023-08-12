import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

weight = torch.randn(1, 60, 80)
#interpolate
# weight = F.interpolate(weight,
#                        scale_factor=8,
#                        mode='linear',
#                        align_corners=True)
weight = weight.unsqueeze(0)
weight = nn.Upsample(scale_factor=8, mode='bilinear',
                     align_corners=True)(weight)
weight = weight.squeeze(0)
print(weight.shape)