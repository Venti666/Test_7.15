import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# 基础模型类，包含日志记录、参数统计等功能
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # 初始化日志记录器，记录模型名称
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        # 强制子类实现 forward 方法
        raise NotImplementedError

    def summary(self):
        # 统计可训练参数的数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'可训练参数数量: {nbr_params}')

    def __str__(self):
        # 打印模型信息和可训练参数数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f"\n可训练参数数量: {nbr_params}"

# 编码器类，包含卷积层和下采样操作
class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        # 定义卷积层序列
        self.down_conv = nn.Sequential(
            # 第一个卷积层，输入通道数为 in_channels，输出通道数为 out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 批量归一化层
            nn.BatchNorm2d(out_channels),
            # ReLU 激活函数
            nn.ReLU(inplace=True),
            # 第二个卷积层，输入和输出通道数均为 out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 最大池化层，用于下采样
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        # 经过卷积层处理
        out = self.down_conv(x)
        # 保存卷积层输出，用于后续的跳跃连接
        out_copy = out
        # 进行下采样
        out_pool = self.pool(out)
        return out_copy, out_pool

# 解码器类，包含反卷积层和上采样操作
class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 反卷积层，用于上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 定义卷积层序列
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        # 进行反卷积上采样
        out = self.up(x)
        if interpolate:
            # 使用双线性插值进行上采样，以获得更好的结果
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True)
        else:
            # 如果上采样后的尺寸与跳跃连接的特征图尺寸不同，进行填充
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # 将跳跃连接的特征图和上采样后的特征图在通道维度上拼接
        out = torch.cat([x_copy, out], dim=1)
        # 经过卷积层处理
        out_conv = self.up_conv(out)
        return out_conv

# 基础 U-Net 模型类
class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=5, freeze_bn=False, **_):
        super(UNet, self).__init__()

        # U-Net 编码器部分
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        # 中间卷积层
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # U-Net 解码器部分
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        # 最终卷积层，输出通道数为类别数
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        # 初始化模型参数
        self._initalize_weights()
        if freeze_bn:
            # 冻结批量归一化层
            self.freeze_bn()

    def _initalize_weights(self):
        # 初始化模型参数
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # 使用 Kaiming 初始化方法初始化卷积层和全连接层的权重
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    # 将偏置项初始化为零
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                # 将批量归一化层的权重初始化为 1，偏置项初始化为零
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        # 编码器部分的前向传播
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        # 中间卷积层的前向传播
        x = self.middle_conv(x)
        # 解码器部分的前向传播
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        # 最终卷积层的前向传播
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # U-Net 没有骨干网络，所有参数从头开始训练
        return []

    def get_decoder_params(self):
        # 返回模型的所有参数
        return self.parameters()

    def freeze_bn(self):
        # 冻结批量归一化层
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()