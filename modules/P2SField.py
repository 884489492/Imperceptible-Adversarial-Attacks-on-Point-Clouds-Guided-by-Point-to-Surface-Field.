import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
    通过P2S场预测梯度来引导攻击
    已经是一个已知的梯度方向，为什么不直接使用它，而是通过 P2SField 去预测一个 pred_grad?
    1、泛化到未知数据，模型通过大量 (noisy_points, target_grad) 学习从噪声点预测表面方向的映射,训练好后，P2S不用指导点就能直接输出梯度
    2、网络通过卷积层（3 -> 64 -> 128 -> 256）和全连接层提取点的几何特征,不仅学习“抵消噪声”，还学习表面梯度指向密度最高的表面。即使噪声模式未知，模型也能根据点云形状预测合理方向
        
"""


class P2SField(nn.Module):
    def __init__(self):
        """
        文章提到是基于深度学习的去噪网络，由于PointNet是处理点云的经典网络，他能捕捉全局和局部的几何信息适合学习表面梯度，
        而且在实验中也攻击了PointNet,选择基于PointNet的变体
        """
        super(P2SField, self).__init__()
        # 输入: 点云 [B, N, 3]
        # 卷积层，提取点云的局部和全局特征
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        # 全连接层，将特征映射到每个点的3D梯度
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 输出梯度 [B, N, 3]

        # 批量归一化，提升训练稳定性
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, points):
        """

        :param points: 带有噪声的点云, [B, N, 3]
        :return: 每个点的梯度 F(q),指向表面
        """

        x = points.transpose(1, 2)  # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, N]
        x = x.transpose(1, 2)  # [B, N, 256]
        x = F.relu(self.fc1(x))  # [B, N, 128]
        x = F.relu(self.fc2(x))  # [B, N, 64]
        x = self.fc3(x)  # [B, N, 3]

        return x  # 输出梯度场 [B, N, 3], 每个点的梯度向量

def train_P2SField(self, P2S_field, data_loader, epoch=50, noise_level=0.5, device="cpu"):
    """
    训练 P2S 场
    :param P2S_field: P2S场模型
    :param data_loader: 干净的点云
    :param epoch:
    :param noise_level: 噪声标准差
    :param device:
    :return:

    P2SField的目标是从noisy_points预测梯度 F(q), 使得 noisy_points + F(q)接近原始的点云
    """
    P2S_field.train()
    optimizer = optim.Adam(P2S_field.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.MSELoss()  # l_2损失

    for epoch in range(epoch):
        total_loss = 0
        for batch_idx, (points, _) in enumerate(data_loader):
            points = points.to(device)  # 干净点云 [B, N, 3]

            # 添加高斯噪声
            noise = torch.randn_like(points, device=device) * noise_level
            noisy_points = points + noise  # 带噪点云

            # 向前传播，预测梯度
            pred_grad = P2S_field(noisy_points)  # 预测梯度 [B, N, 3]

            # 目标梯度: 从噪声点到原始点的方向
            """
            target_grad = points - noisy = points - (points + npise) = -noise

            P2SField的目标是从带噪声的点云中预测梯度，使得噪声点 q 沿着F(q)移动后依然接近表面 S, 
            F(q)是表面梯度，指向表面方向,即从当前点(noisy_points)指向目标点(points或表面 S)
            模型预测的梯度 pred_grad被训练地接近 target_grad: 

            noisy_point + pred = (points + noise) + (-noise) = point, 噪声点加上预测梯度后回到原来的位置

            """
            target_grad = points - noisy_points  # -noise
            loss = criterion(pred_grad, target_grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1} / {epoch}, Average Loss: {avg_loss: .6f}")

    torch.save(P2S_field.state_dict(), 'P2S.pth')
    print("P2S训练完成,并保存在 P2S.pth")


