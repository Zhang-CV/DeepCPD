import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.reshape(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).reshape(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.reshape(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    x = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN(nn.Module):
    def __init__(self, args, d_model):
        super(DGCNN, self).__init__()
        self.d_model = d_model
        self.cuda = args.cuda

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(256, self.d_model, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm1d(self.d_model)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = get_graph_feature(x.transpose(2, 1))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2)
        x = self.relu(self.bn5(self.conv5(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        out = self.relu(self.bn6(self.conv6(x)))

        return out.squeeze(-1)

