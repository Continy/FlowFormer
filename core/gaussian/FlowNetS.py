import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from submodules import *
from torch.distributions import MultivariateNormal, MixtureSameFamily


class MixtureGaussianConv(nn.Module):

    def __init__(self, cfg):
        super(MixtureGaussianConv, self).__init__()
        mixture_num = cfg.mixture_num
        self.mixture_num = mixture_num
        #self.means_layer = FlowNetS(input_channels=24, mixture_num=1)
        self.vars_layer = FlowNetS(input_channels=24, mixture_num=mixture_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        #means = self.means_layer(x)
        vars = self.relu(self.vars_layer(x)) + 1e-6

        return vars


class FlowNetS(nn.Module):

    def __init__(self, input_channels=12, mixture_num=1, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm,
                          input_channels,
                          64,
                          kernel_size=7,
                          stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)

        self.deconv2 = deconv(256, 64)
        self.deconv1 = deconv(128, 32)
        self.predict_flow3 = predict(256, mixture_num)
        self.predict_flow2 = predict(192 + mixture_num * 2, mixture_num)
        self.predict_flow1 = predict(96 + mixture_num * 2, mixture_num)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(mixture_num * 2,
                                                       mixture_num * 2,
                                                       4,
                                                       2,
                                                       1,
                                                       bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(mixture_num * 2,
                                                       mixture_num * 2,
                                                       4,
                                                       2,
                                                       1,
                                                       bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        print(out_conv2.shape)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        print(out_conv3.shape)
        flow3 = self.predict_flow3(out_conv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)

        out_deconv2 = self.deconv2(out_conv3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)
        flow2_up = self.upsampled_flow2_to_1(flow2)

        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)

        flow1 = self.predict_flow1(concat1)
        flow1_up = self.upsample1(flow1)
        return flow1_up


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    x = torch.randn(1, 24, 480, 640)
    mixture_num = 3
    model = FlowNetS(input_channels=24, mixture_num=mixture_num)
    # forward
    vars = torch.exp(model(x) + 1e-6)
    #split vars into mixture_num parts
    vars = vars.split(vars.shape[1] // mixture_num, dim=1)
    print(f"vars: {vars[1].shape}")
    print(f"Parameter Count: {count_parameters(model)}")