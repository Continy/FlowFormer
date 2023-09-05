import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from submodules import *


class FlowNetS(nn.Module):

    def __init__(self, input_channels=12, batchNorm=True):
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
        self.predict_flow3 = predict_flow(256)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(98)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2,
                                                       2,
                                                       4,
                                                       2,
                                                       1,
                                                       bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2,
                                                       2,
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
        print(out_conv1.shape)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        flow3 = self.predict_flow3(out_conv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)

        out_deconv2 = self.deconv2(out_conv3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        print(out_conv2.shape)
        print(out_deconv2.shape)
        print(flow3_up.shape)
        flow2 = self.predict_flow2(concat2)
        flow2_up = self.upsampled_flow2_to_1(flow2)

        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)

        flow1 = self.predict_flow1(concat1)
        flow1_up = self.upsample1(flow1)
        return flow1_up


if __name__ == '__main__':
    # 定义输入张量
    x = torch.randn(1, 12, 480, 640)
    # 创建 FlowNetS 类的实例
    model = FlowNetS(input_channels=12)
    # 调用 forward 方法
    vars = model(x)

    print(f"x: {x.shape}")
    print(f"vars: {vars.shape}")