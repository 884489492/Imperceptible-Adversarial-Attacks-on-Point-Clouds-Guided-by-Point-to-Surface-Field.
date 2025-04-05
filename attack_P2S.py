import torch
import torch.nn as nn

def P2S_guided_attack(model, P2SField, points, labels, epsilon=0.05,
                      lambda1=0.1,iterations=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """

    :param model: 目标 3D网络分类器
    :param P2SField: P2S场
    :param points: 干净的点云 [B, N, 3]
    :param labels: 正确的标签 [B]
    :param epsilon: 扰动步长
    :param iterations:迭代次数
    :param device:
    :param lambda1: 约束项权重
    :return:
    """
    points = points.to(device)
    labels = labels.to(device)
    # 随机初始化扰动
    adv_point = perturb(epsilon, points)
    # 迭代攻击
    adv_point = attack(P2SField, adv_point, epsilon, iterations, labels, lambda1, model, points)
    return adv_point


def perturb(epsilon, points):
    perturb_init = torch.randn_like(points) * epsilon
    adv_point = points + perturb_init
    adv_point = adv_point.detach().requires_grad_(True)
    return adv_point


def attack(P2SField, adv_point, epsilon, iterations, labels, lambda1, model, points):
    for t in range(iterations):
        # 使用IFGM来生成开始的扰动
        logits = model(adv_point)
        mis_loss = nn.CrossEntropyLoss()(logits, labels)  # 计算交叉熵损失，衡量模型输出与真实标签的差异
        c_loss = torch.mean((points - adv_point) ** 2)  # 约束损失
        total_loss = mis_loss + lambda1 * c_loss

        model.zero_grad()
        total_loss.backward()
        grad = adv_point.grad.data
        perturb = epsilon * grad.sign()  # 生成扰动张量perturb，沿梯度的符号方向移动
        adv_point_new = adv_point + perturb

        #  用P2S场调整扰动方向
        P2S_adjust = P2SField(adv_point_new)
        # 与表面的距离调整,估算点到原始点的距离
        dist = torch.norm(adv_point_new - points, dim=-1, keepdim=True)
        # 调整 F(q) 大小，远处的点调整更大
        scale = torch.tanh(dist)
        P2S_adjust = scale * P2S_adjust
        adv_point_new = adv_point_new + P2S_adjust

        # 限制有效范围
        adv_point_new = torch.clamp(adv_point_new, -1, 1)

        adv_point = adv_point_new.detach().requires_grad_(True)
    return adv_point



