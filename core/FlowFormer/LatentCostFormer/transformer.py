import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from utils.utils import coords_grid, bilinear_sampler, upflow8
from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from ..encoders import twins_svt_large_context, twins_svt_large
from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder


class FlowFormer(nn.Module):

    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg['latentcostformer']
        self.mode = cfg.training_mode
        self.memory_encoder = MemoryEncoder(self.cfg)
        self.memory_decoder = MemoryDecoder(self.cfg)
        if self.cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(
                pretrained=self.cfg.pretrain)
        elif self.cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256,
                                                norm_fn='instance')

    def forward(self, image1, image2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions, vars = self.memory_decoder(cost_memory,
                                                     context,
                                                     data,
                                                     flow_init=flow_init,
                                                     mode=self.mode)

        return flow_predictions, vars
