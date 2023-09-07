import loguru
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np

from utils.utils import coords_grid, bilinear_sampler, upflow8
from einops.layers.torch import Rearrange
from einops import rearrange

from ..encoders import twins_svt_large

import time
import sys
from timm.models.layers import Mlp, DropPath, activations, to_2tuple, trunc_normal_
from .swin_transformer_unet_skip_expand_decoder_sys import BasicLayer, PatchEmbed, PatchExpand, FinalPatchExpand_X4, BasicLayer_up, PatchMerging
from ...corr import CorrBlock


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups,
                                      num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups,
                                      num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups,
                                          num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):

    def __init__(self,
                 input_dim=3,
                 output_dim=128,
                 norm_fn='batch',
                 dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        mul = input_dim // 3

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64 * mul)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64 * mul)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64 * mul)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim,
                               64 * mul,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64 * mul
        self.layer1 = self._make_layer(64 * mul, stride=1)
        self.layer2 = self._make_layer(96 * mul, stride=2)
        self.layer3 = self._make_layer(128 * mul, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128 * mul, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes,
                               dim,
                               self.norm_fn,
                               stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num += np.prod(param.size())

        return num

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class MemoryEncoder(nn.Module):

    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg

        if cfg.fnet == 'twins':
            self.feat_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            self.feat_encoder = BasicEncoder(output_dim=256,
                                             norm_fn='instance')
        else:
            exit()
        self.channel_convertor = nn.Conv2d(cfg.encoder_latent_dim,
                                           cfg.encoder_latent_dim,
                                           1,
                                           padding=0,
                                           bias=False)

    def corr(self, fmap1, fmap2):

        batch, dim, ht, wd = fmap1.shape
        fmap1 = rearrange(fmap1,
                          'b (heads d) h w -> b heads (h w) d',
                          heads=self.cfg.cost_heads_num)
        fmap2 = rearrange(fmap2,
                          'b (heads d) h w -> b heads (h w) d',
                          heads=self.cfg.cost_heads_num)
        corr = einsum('bhid, bhjd -> bhij', fmap1, fmap2)
        corr = corr.permute(0, 2, 1, 3).view(batch * ht * wd,
                                             self.cfg.cost_heads_num, ht, wd)
        #corr = self.norm(self.relu(corr))
        corr = corr.view(batch, ht * wd, self.cfg.cost_heads_num,
                         ht * wd).permute(0, 2, 1, 3)
        corr = corr.view(batch, self.cfg.cost_heads_num, ht, wd, ht, wd)

        return corr

    def forward(self, img1, img2, data, context=None):
        # The original implementation
        # feat_s = self.feat_encoder(img1)
        # feat_t = self.feat_encoder(img2)
        # feat_s = self.channel_convertor(feat_s)
        # feat_t = self.channel_convertor(feat_t)

        imgs = torch.cat([img1, img2], dim=0)
        feats = self.feat_encoder(imgs)
        feats = self.channel_convertor(feats)
        B = feats.shape[0] // 2

        feat_s = feats[:B]
        feat_t = feats[B:]

        B, C, H, W = feat_s.shape
        size = (H, W)

        if self.cfg.feat_cross_attn:
            feat_s = feat_s.flatten(2).transpose(1, 2)
            feat_t = feat_t.flatten(2).transpose(1, 2)

            for layer in self.layers:
                feat_s, feat_t = layer(feat_s, feat_t, size)

            feat_s = feat_s.reshape(B, *size, -1).permute(0, 3, 1,
                                                          2).contiguous()
            feat_t = feat_t.reshape(B, *size, -1).permute(0, 3, 1,
                                                          2).contiguous()

        cost_volume = self.corr(feat_s, feat_t)
        B, heads, H1, W1, H2, W2 = cost_volume.shape
        cost_maps = cost_volume.permute(0, 2, 3, 1, 4, 5).contiguous().view(
            B * H1 * W1, self.cfg.cost_heads_num, H2, W2)
        print(cost_maps.shape)
        return cost_maps


class UNet(nn.Module):

    def __init__(self,
                 img_size=[240, 320],
                 patch_size=2,
                 in_chans=6,
                 num_classes=19200,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[1, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                 window_size=5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 final_upsample="expand_first",
                 cfg=None,
                 **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}"
            .format(depths, depths_decoder, drop_path_rate, num_classes))
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        #print('patches_resolution:', patches_resolution)

        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2**(self.num_layers - 1 - i_layer)),
                int(embed_dim *
                    2**(self.num_layers - 1 -
                        i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] //
                                      (2**(self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] //
                                      (2**(self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2**(self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2**(self.num_layers - 1 - i_layer)),
                    input_resolution=(patches_resolution[0] //
                                      (2**(self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] //
                                      (2**(self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(
                        self.num_layers - 1 - i_layer
                    )]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if
                    (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size[0] // patch_size,
                                  img_size[1] // patch_size),
                dim_scale=4,
                dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,
                                    out_channels=self.num_classes,
                                    kernel_size=1,
                                    bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            print(x.shape)
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  #B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.num_classes
        return flops