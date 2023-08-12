from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from MMG import MultivariateMixtureGaussian as MMG
from MMG_loss import MMGLoss

gamma = 0.75
#i_weight = gamma**(n_predictions - i - 1)
#loss = i_weight * (valid[:, None] * i_loss).mean()

#Our new method of calculating the loss:
#estimate i_weight by gaussian distribution
n_predictions = 12
sum = 0
for i in range(n_predictions):
    i_weight = gamma**(n_predictions - i - 1)
    sum += i_weight
print(sum)


#input:prediction index&feature map
class WeightUpdateBlock(nn.Module):

    def __init__(self, hidden_size=128, mixture_num=3, para_num=1):
        self.mixture_num = mixture_num
        self.para_num = para_num
        self.weights = MMG(hidden_size, mixture_num, para_num)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net):
        dist_range = [-1, 1]

        mask = .25 * self.mask(net)  #mask.shape torch.Size([N, 576, 60, 80])
        return mask
