import torch
import numpy as np
from torch.distributions import MultivariateNormal, MixtureSameFamily


class UMGLoss():

    def gaussian_pdf(self, y, mean, var, eps=1e-8):

        pdf = 1 / (torch.sqrt(2 * np.pi * var)) * torch.exp(-(y - mean)**2 /
                                                            (2 * var))

        return pdf

    def category_loss(self, x_theta, label_r, label_theta):
        #计算极坐标的损失函数
        loss = torch.mean((x_theta - label_theta)**2)
        return loss

    # 定义损失函数
    def mdn_loss(self, means, vars, weights, target):
        pdfs = self.gaussian_pdf(target, means, vars)
        pdfs_weighted = torch.sum(pdfs * weights, dim=1)
        loss = -torch.log(pdfs_weighted + 1e-6)
        loss = torch.mean(loss)
        return loss

    def total_loss(self, value_mean, value_var, value_weight, angle_mean,
                   angle_var, angle_weight, value, angle):

        value_loss = self.mdn_loss(value_mean, value_var, value_weight, value)

        angle_loss = self.mdn_loss(angle_mean, angle_var, angle_weight, angle)
        loss = value_loss + 0.001 * angle_loss

        loss = torch.mean(loss)
        return loss


class MMGLoss():

    # 定义损失函数
    def total_loss(self, means, covs, weights, target):
        #将covs转换为协方差矩阵
        covs = torch.diag_embed(covs)
        mvn = MultivariateNormal(means, covs)
        dist = MixtureSameFamily(
            torch.distributions.Categorical(probs=weights), mvn)
        pdf = dist.log_prob(target)
        pdf = torch.exp(pdf)
        loss = -torch.log(pdf + 1e-6)
        loss = torch.mean(loss)
        return loss
