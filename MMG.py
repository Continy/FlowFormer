import torch
import torch.nn as nn


class MultivariateMixtureGaussian(nn.Module):

    def __init__(self, hidden_size=32, mixture_num=3, para_num=2):
        super(MultivariateMixtureGaussian, self).__init__()
        self.mixture_num = mixture_num
        self.para_num = para_num
        self.fc1 = nn.Linear(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.mean_encoder = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.cov_encoder = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.mean_decoder = nn.Linear(hidden_size, mixture_num * para_num)
        self.cov_decoder = nn.Linear(hidden_size, mixture_num * para_num)
        self.bn5 = nn.BatchNorm1d(hidden_size * 3)
        self.weight = nn.Linear(hidden_size * 3, mixture_num)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.bn1(x))
        x = torch.relu(self.bn2(self.fc2(x)))
        mean = torch.relu(self.bn3(self.mean_encoder(x)))
        cov = torch.relu(self.bn4(self.cov_encoder(x)))
        y = torch.cat((mean, cov, x), dim=1)
        y = self.bn5(y)
        gaussuan_weight = torch.softmax(self.weight(y) + 1e-6, dim=1)
        mean = self.mean_decoder(mean)
        cov = torch.exp(self.cov_decoder(cov) + 1e-6)
        mean = mean.view(-1, self.mixture_num, self.para_num)
        cov = cov.view(-1, self.mixture_num, self.para_num)
        return mean, cov, gaussuan_weight
