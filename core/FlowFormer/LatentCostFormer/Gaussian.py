import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily


class MixtureGaussianConv(nn.Module):

    def __init__(self, cfg):
        super(MixtureGaussianConv, self).__init__()
        hidden_size = cfg.hidden_size
        mixture_num = cfg.mixture_num

        self.mixture_num = mixture_num

        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, hidden_size, 1, padding=0),
                                  nn.ReLU(inplace=True))
        self.cov_encoder = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.mean_decoder = nn.Conv2d(hidden_size, mixture_num, 3, padding=1)
        self.cov_decoder = nn.Conv2d(hidden_size, mixture_num, 3, padding=1)

        self.cov_encoder_bn = nn.BatchNorm2d(hidden_size)
        self.cov_decoder_bn = nn.BatchNorm2d(mixture_num)
        self.weight = nn.Conv2d(hidden_size, mixture_num, 1, padding=0)

    def forward(self, x):
        x = self.mask(x)
        N, C, H, W = x.size()

        cov = self.cov_encoder(x)
        cov = F.relu(cov)
        cov = self.cov_encoder_bn(cov)
        cov = self.cov_decoder(cov)
        cov = self.cov_decoder_bn(cov)
        cov = torch.exp(cov + 1e-6)
        cov = cov.view(N, self.mixture_num, self.para_num, H, W)

        return cov


def gaussian_loss(means, covs, weights, target):
    #将covs转换为协方差矩阵
    covs = torch.diag_embed(covs)
    mvn = MultivariateNormal(means, covs)
    dist = MixtureSameFamily(torch.distributions.Categorical(probs=weights),
                             mvn)
    pdf = dist.log_prob(target)
    pdf = torch.exp(pdf)
    loss = -torch.log(pdf + 1e-6)
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    # 定义输入张量
    x = torch.randn(1, 128, 60, 80)
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.hidden_size = 128
    cfg.mixture_num = 5
    cfg.para_num = 1
    # 创建 MixtureGaussianConv 类的实例
    model = MixtureGaussianConv(cfg)

    # 调用 forward 方法
    vars = model(x)

    print(f"x: {x.shape}")

    predictions = []
    for i in range(12):
        predictions.append(torch.randn(1, 2, 60, 80))
    #新增一个维度，将predictions变成一个tensor,shape为(1,12,2,60,80)
    predictions = torch.stack(predictions, dim=1)
    #将means&vars&weights变成(1,60，80,5,2)

    vars = vars.permute(0, 3, 4, 1, 2)

    vars = torch.diag_embed(vars)

    print(f"cov: {vars.shape}")

    dist = MultivariateNormal(means, vars)

    dist = MixtureSameFamily(torch.distributions.Categorical(probs=weights),
                             dist)
    # target = torch.randn(1, 60, 80, 1)
    # pdf = dist.log_prob(target)
    # pdf = torch.exp(pdf)
    # print(f"pdf: {pdf.shape}")
    n_predictions = 12
    lable = torch.linspace(-5, 5, n_predictions).to(weights.device)
    print(f"target: {lable}")
    result = []
    pdfs = []
    for i in range(n_predictions):
        #将lable[i]变成(1,60,80,1)
        target = lable[i].expand_as(torch.zeros(1, 60, 80, 1))
        print(f"target: {target.shape}")
        pdfs.append(dist.log_prob(target))
    pdfs = torch.stack(pdfs, dim=3)
    print(f"pdfs: {pdfs.shape}")