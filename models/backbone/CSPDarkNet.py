import torch
import torch.nn as nn
class CSPDarknet(nn.Module):
    def __init__(self, num_classes):
        super(CSPDarknet, self).__init__()
        
        # 定义输入卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # CSPDarknet的主干网络部分
        self.stage1 = self.make_stage(32, 64, 1)
        self.stage2 = self.make_stage(64, 128, 2)
        self.stage3 = self.make_stage(128, 256, 8)
        self.stage4 = self.make_stage(256, 512, 8)
        self.stage5 = self.make_stage(512, 1024, 4)

        # 添加分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def make_stage(self, in_channels, out_channels, num_blocks):
        # 创建CSPDarknet的阶段
        stage = []

        # 第一个残差块，特殊处理
        stage.append(nn.Conv2d(in_channels, out_channels // 2, kernel_size=1))
        stage.append(nn.BatchNorm2d(out_channels // 2))
        stage.append(nn.LeakyReLU(0.1))
        stage.append(nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1))
        stage.append(nn.BatchNorm2d(out_channels))
        stage.append(nn.LeakyReLU(0.1))
        stage.append(nn.Conv2d(out_channels, out_channels // 2, kernel_size=1))
        stage.append(nn.BatchNorm2d(out_channels // 2))
        stage.append(nn.LeakyReLU(0.1))

        # 创建多个残差块
        for _ in range(num_blocks):
            stage.append(nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1))
            stage.append(nn.BatchNorm2d(out_channels // 2))
            stage.append(nn.LeakyReLU(0.1))
            stage.append(nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1))
            stage.append(nn.BatchNorm2d(out_channels))
            stage.append(nn.LeakyReLU(0.1))
            stage.append(nn.Conv2d(out_channels, out_channels // 2, kernel_size=1))
            stage.append(nn.BatchNorm2d(out_channels // 2))
            stage.append(nn.LeakyReLU(0.1))

        return nn.Sequential(*stage)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.classifier(x)
        return x
