import torch
import torch.nn as nn

def calculate_ndvi(image):
    """
    计算 NDVI 指数
    :param image: 输入图像，形状为 (batch_size, channels, height, width)
    :return: NDVI 指数，形状为 (batch_size, height, width)
    """
    red_band = image[:, 3, :, :]  # 红波段是第 4 个通道
    nir_band = image[:, 4, :, :]  # 近红外波段是第 5 个通道
    denominator = nir_band + red_band
    mask = denominator != 0  # 避免除零错误
    ndvi = torch.zeros_like(red_band, dtype=torch.float32)
    ndvi[mask] = (nir_band[mask] - red_band[mask]) / denominator[mask]
    # 裁剪异常值（理论 NDVI 范围为 [-1, 1]）
    ndvi = torch.clamp(ndvi, -1, 1)
    return ndvi

class CustomNDVILoss(nn.Module):
    def __init__(self, alpha=2.0):
        """
        初始化自定义 NDVI 损失函数
        :param alpha: 对 NDVI 在 [0, 0.5] 范围内的像素的权重因子
        """
        super(CustomNDVILoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha

    def forward(self, output, target, image):
        """
        前向传播计算损失
        :param output: 模型的输出，形状为 (batch_size, num_classes, height, width)
        :param target: 真实标签，形状为 (batch_size, height, width)
        :param image: 输入图像，形状为 (batch_size, channels, height, width)
        :return: 自定义损失值
        """
        # 计算 NDVI 指数
        ndvi = calculate_ndvi(image)
        # 计算交叉熵损失
        ce_loss = self.criterion(output, target)
        # 创建权重掩码
        weight_mask = torch.ones_like(ce_loss)
        # 对 NDVI 在 [0, 0.5] 范围内的像素增加权重
        mask = (ndvi >= 0) & (ndvi <= 0.5)
        weight_mask[mask] = self.alpha
        # 应用权重掩码
        weighted_loss = ce_loss * weight_mask
        # 计算最终损失
        final_loss = weighted_loss.mean()
        return final_loss