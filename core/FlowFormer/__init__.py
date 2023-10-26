import torch


def build_flowformer(cfg):
    name = cfg.transformer

    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")

    return FlowFormer(cfg)


def build_gaussian(cfg):
    weight = cfg.weight
    from gaussian.FlowNetS import MixtureGaussianConv
    from core.FlowFormer.LatentCostFormer.dimension_test import UNet
    if cfg[weight].method == 'U-net':
        return UNet()
    if cfg[weight].method == 'FlowNetS':
        return MixtureGaussianConv(cfg[weight])
    else:
        raise ValueError(
            f"FlowFormer = {weight.method} is not a valid architecture!")
