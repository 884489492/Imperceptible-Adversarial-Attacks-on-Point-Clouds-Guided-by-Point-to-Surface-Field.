import os
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np
from sklearn.model_selection import train_test_split
"""
获取并处理数据
"""

class ModelNet40Dataset(Dataset):
    def __init__(self, data_list, label_list, num_points=1024):
        self.data_list = data_list    # 文件路径列表
        self.label_list = label_list  # 对应的标签列表
        self.num_points = num_points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        加载 .obj 文件并采样点云
        :param idx:
        :return:
        """
        mesh = trimesh.load(self.data_list[idx])
        points = mesh.sample(self.num_points)   # 从一个 3D 网格（mesh）中采样固定数量点（self.num_points）的过程，用于生成点云数据
        points = (points - points.mean(axis=0)) / np.max(np.linalg.norm(points, axis=1))    # 中心化、归一化
        label = self.label_list[idx]
        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def loadModelnet40(data_dir, num_points=1024, train_split=0.8):    # 加载并分割数据
    data_list = []
    label_list = []
    class_map = {}

    # 获取类别列表
    classes = sorted(os.listdir(data_dir))
    for idx, cls in enumerate(classes):
        class_map[cls] = idx
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            for file in os.listdir(cls_dir):
                if file.endswith('.obj'):
                    data_list.append(os.path.join(cls_dir, file))
                    label_list.append(idx)
    # 分割训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(data_list, label_list,
                                                                        train_size=train_split,
                                                                        stratify=label_list,
                                                                        random_state=42)
    train_data = ModelNet40Dataset(train_data, train_labels, num_points)
    test_data = ModelNet40Dataset(test_data, test_labels,num_points)
    return train_data, test_data, class_map

