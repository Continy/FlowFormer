import torch


def build_flowformer(cfg):
    name = cfg.transformer

    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")

    return FlowFormer(cfg[name])


def build_gaussian(cfg):
    weight = cfg.weight
    if weight == 'mixturegaussian':
        from .LatentCostFormer.Gaussian import MixtureGaussianConv
    return MixtureGaussianConv(cfg[weight])