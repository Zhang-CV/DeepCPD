import os
import numpy as np
import torch
import open3d as o3d
import h5py
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


# source point cloud for ICP
def refinement_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(base_dir, 'data', 'GPM_r.ply')
    pcd = o3d.io.read_point_cloud(file)
    pcd = np.asarray(pcd.points)
    pcd = np.random.permutation(pcd)
    return pcd


def load_data(partition, scalar):
    # read source point clouds
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(base_dir, 'data', 'GPM.ply')
    pcd = o3d.io.read_point_cloud(file)
    pcd = np.asarray(pcd.points)
    # pcd = pcd / np.max(abs(pcd))
    pcd = np.expand_dims(pcd, axis=0)
    pcd = np.random.permutation(pcd)
    pc_sources = 0.
    if partition == 'train':
        pc_sources = np.repeat(pcd, 8192, axis=0)
    elif partition == 'eval':
        pc_sources = np.repeat(pcd, 512, axis=0)
    elif partition == 'test':
        pc_sources = np.repeat(pcd, 4096, axis=0)

    # read target point cloud
    files = []
    with open(os.path.join(base_dir, 'data', '%s_data' % partition, '%s_data.txt' % partition), "r") as f:
        for file in f.readlines():
            file = file.strip('\n')
            files.append(file)
    pc_means = []
    pcd_targets = []
    flag = 0
    for file in files:
        file = os.path.join(base_dir, 'data', '%s_data' % partition, file)
        pcd = o3d.io.read_point_cloud(file)
        pcd = np.asarray(pcd.points) * 7.3951
        pcd = np.expand_dims(pcd, axis=0)
        pcd_mean = pcd.mean(axis=1, keepdims=False)
        pc_means.append(pcd_mean)
        pcd = pcd - pcd_mean
        pcd = np.random.permutation(pcd)
        pcd_targets.append(pcd)
        flag+=1
    # read R and T
    pc_means = np.asarray(pc_means) / scalar
    h5_name = os.path.join(base_dir, 'data', '%s_data' % partition, 'transform.h5')
    f = h5py.File(h5_name, 'r')
    angles = f['rotation'][:].astype('float32')
    translations = f['translation'][:].astype('float32')
    f.close()
    angles = angles.T[:len(pcd_targets)]
    translations = translations.T[:len(pcd_targets)] - pc_means.squeeze(1)
    # pcd_targets = np.concatenate(pcd_targets)

    return pc_sources, pcd_targets, angles, translations


class Satellite(Dataset):
    def __init__(self, partition='train', scalar=1., gaussian_noise=False):
        self.data_sources, self.data_targets, self.angles, self.translations = load_data(partition, scalar)
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.scalar = scalar

    def __getitem__(self, item):
        pts_src = self.data_sources[item]
        pts_target = self.data_targets[item] / self.scalar
        pts_target = np.asarray(pts_target).squeeze(0)
        angle = self.angles[item]
        translation_ab = self.translations[item]

        rotation_ab = Rotation.from_euler('XYZ', angle, degrees=True)
        rotation_ab = np.asarray(rotation_ab.as_matrix()).transpose()
        translation_ab = np.expand_dims(translation_ab, axis=0)
        # return source pointcloud, target pointcloud (1, num_points, 3)
        return pts_src.astype('float32'), pts_target.astype('float32'), rotation_ab.astype('float32'), \
               translation_ab.astype('float32')

    def __len__(self):
        return len(self.data_targets)
