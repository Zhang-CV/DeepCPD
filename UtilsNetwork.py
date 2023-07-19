import torch
import time
import open3d as o3d
from torch import nn
import numpy as np


# Reshape layer
class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


def gmm_register(mu_src, sigma_src, gamma_target, pts_target):
    '''
        Inputs:
            gamma: B x N x J
            pts: B x N x 3
    '''
    Np_target = pts_target.size(1)
    # mu_src: B x J × 3
    # c_src : B x 1 × 3
    c_src = torch.sum(gamma_target @ mu_src, dim=1, keepdim=True) / Np_target
    # c_target : B x 1 × 3
    c_target = torch.sum(gamma_target.transpose(1, 2) @ pts_target, dim=1, keepdim=True) / Np_target
    # A = (pts_target - c_target).transpose(1, 2) @ gamma_target @ (mu_src - c_src)
    A = torch.sum(((pts_target - c_target).transpose(1, 2) @ gamma_target @ (mu_src - c_src)).unsqueeze(1)
                  @ sigma_src.inverse(), dim=1)
    U, _, V = torch.svd(A)
    U = U
    V = V
    C = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    C[:, 2, 2] = torch.det(U @ V.transpose(1, 2))
    R = U @ C @ V.transpose(1, 2)
    # s = torch.trace(A.transpose(1, 2) @ R) / torch.trace(c_src.transpose(1, 2) @ c_src)
    t = c_target.transpose(1, 2) - R @ c_src.transpose(1, 2)
    return R, t.transpose(1, 2)


def gmm_params(gamma, pts):
    '''
    Inputs:
        gamma: B x N x J
        pts: B x N x 3
    '''
    # pi: B x J
    pi = gamma.mean(dim=1)
    # Npi: B × J
    Npi = pi * gamma.shape[1]
    # mu: B x J x 3
    mu = gamma.transpose(1, 2) @ pts / Npi.unsqueeze(2)
    # diff: B x N x J x 3
    diff = pts.unsqueeze(2) - mu.unsqueeze(1)
    # sigma: B x J x 3 x 3
    eye = torch.eye(3).unsqueeze(0).unsqueeze(1).to(gamma.device)
    sigma = ((diff.pow(2) * gamma.unsqueeze(3)).sum(3, True).repeat(1, 1, 1, 3).sum(dim=1) / Npi.unsqueeze(-1))
    sigma = sigma.unsqueeze(3) * eye
    return pi, mu, sigma


def ICP(src, target):
    src = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(src))
    target = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target))
    src.estimate_normals()

    threshold = 1
    trans_init = torch.eye(4, 4)
    start = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        target, src, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    end = time.time()
    transformation = reg_p2p.transformation
    rotation_ab_icp = transformation[0: 3][:, 0: 3].T
    translation_ba_icp = transformation[0: 3][:, 3:4].transpose(1, 0)
    translation_ab_icp = -rotation_ab_icp @ translation_ba_icp.transpose(1, 0)
    return rotation_ab_icp, translation_ab_icp.transpose(1, 0), end - start


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

