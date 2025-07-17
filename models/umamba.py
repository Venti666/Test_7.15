import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'可训练参数数量: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f"\n可训练参数数量: {nbr_params}"


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        out = self.down_conv(x)
        out_copy = out
        out_pool = self.pool(out)
        return out_copy, out_pool


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True)
        else:
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv


# 修正后的 Mamba 层（严格保持尺寸不变）
class MambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 输入通道数
        # 仅使用不改变尺寸的操作：LayerNorm + 1x1 卷积（替代可能改变长度的 1D 卷积）
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1 卷积，不改变空间尺寸

    def forward(self, x):
        B, C, H, W = x.shape  # (batch, channels, height, width)
        assert C == self.dim, f"输入通道数 {C} 与 Mamba 层维度 {self.dim} 不匹配"
        
        # 展平空间维度：(B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, L, C)，L=H*W
        x_norm = self.norm(x_flat)  # LayerNorm 作用于通道维度
        
        # 恢复空间维度后用 1x1 卷积（确保尺寸不变）
        x_feat = x_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_conv = self.conv(x_feat)  # (B, C, H, W)，尺寸不变
        
        # 残差连接：确保输出与输入尺寸完全一致
        return x + x_conv


class UMamba(BaseModel):
    def __init__(self, num_classes, in_channels=5, freeze_bn=False, **_):
        super(UMamba, self).__init__()

        # 编码器
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        
        # 中间卷积与 Mamba 层（确保输入输出尺寸一致）
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mamba_layer = MambaLayer(dim=1024)  # 维度与中间层通道数一致
        
        # 解码器
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self._initalize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        # 编码器前向传播
        x1, x = self.down1(x)       # x1: (B, 64, 256, 256), x: (B, 64, 128, 128)
        x2, x = self.down2(x)       # x2: (B, 128, 128, 128), x: (B, 128, 64, 64)
        x3, x = self.down3(x)       # x3: (B, 256, 64, 64), x: (B, 256, 32, 32)
        x4, x = self.down4(x)       # x4: (B, 512, 32, 32), x: (B, 512, 16, 16)
        
        # 中间层 + Mamba 层（尺寸严格不变）
        x = self.middle_conv(x)     # (B, 1024, 16, 16)
        x = self.mamba_layer(x)     # (B, 1024, 16, 16)（与输入尺寸完全一致）
        
        # 解码器前向传播
        x = self.up1(x4, x)         # (B, 512, 32, 32)
        x = self.up2(x3, x)         # (B, 256, 64, 64)
        x = self.up3(x2, x)         # (B, 128, 128, 128)
        x = self.up4(x1, x)         # (B, 64, 256, 256)
        
        # 最终输出
        x = self.final_conv(x)      # (B, num_classes, 256, 256)
        return x


# 测试代码
if __name__ == "__main__":
    num_classes = 10
    model = UMamba(num_classes=num_classes, in_channels=5)
    input_tensor = torch.randn(1, 5, 256, 256)  # 输入形状：(1, 5, 256, 256)
    output = model(input_tensor)
    print(f"输出形状: {output.shape}")  # 预期输出：(1, 10, 256, 256)