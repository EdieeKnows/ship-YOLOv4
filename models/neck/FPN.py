import torch
import torch.nn as nn
import torch.nn.functional as F

# FPN的neck部分
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        # 创建FPN的lateral连接层和output层
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

    def forward(self, feature_maps):
        # 构建FPN的前向传播逻辑
        pyramid_features = []

        # 从底层到高层计算lateral连接
        for i, feature in enumerate(reversed(feature_maps)):
            if i == 0:
                lateral_feature = self.lateral_convs[i](feature)
                pyramid_features.insert(0, lateral_feature)
            else:
                lateral_feature = self.lateral_convs[i](feature)
                lateral_feature = lateral_feature + F.interpolate(pyramid_features[0], scale_factor=2, mode="nearest")
                pyramid_features.insert(0, lateral_feature)

        # 计算output特征
        output_features = [self.output_convs[i](pyramid_features[i]) for i in range(len(pyramid_features))]

        return output_features

# 创建CSPDarknet模型
num_classes = 80  # 根据你的数据集设置类别数
cspdarknet = CSPDarknet(num_classes)

# 创建FPN模型并将其与CSPDarknet连接
in_channels_list = [512, 256, 128]  # 根据CSPDarknet的输出通道数设置
fpn = FPN(in_channels_list, out_channels=256)

# 连接CSPDarknet和FPN
class CSPDarknetWithFPN(nn.Module):
    def __init__(self, cspdarknet, fpn):
        super(CSPDarknetWithFPN, self).__init__()
        self.cspdarknet = cspdarknet
        self.fpn = fpn

    def forward(self, x):
        feature_maps = self.cspdarknet(x)
        fpn_features = self.fpn(feature_maps)
        return fpn_features

# 创建包含CSPDarknet和FPN的模型
model = CSPDarknetWithFPN(cspdarknet, fpn)

# 打印模型结构
print(model)