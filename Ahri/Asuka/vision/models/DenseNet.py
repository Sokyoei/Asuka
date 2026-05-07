"""
DenseNet

SeeAlso:
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [torchvision.models.densenet](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)
"""

import torch
from torch import Tensor, nn

from Ahri.Asuka.nn import Conv


class DenseLayer(nn.Module):

    def __init__(self, in_channels: int, growth_rate: float):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # 1x1 卷积压缩通道
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        # 3x3 卷积增加特征
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        # 拼接输入和输出
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, num_layers: int, in_channels: int, growth_rate: float):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransitionLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        growth_rate: float = 32,
        block_config: tuple = (6, 12, 24, 16),
        theta: float = 0.5,
    ):
        super(DenseNet, self).__init__()

        # 初始卷积层
        num_init_features = 2 * growth_rate
        self.conv1 = Conv(3, num_init_features, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # DenseNet 的不同阶段
        channels = num_init_features

        self.denseblock1 = DenseBlock(block_config[0], channels, growth_rate)
        channels += block_config[0] * growth_rate

        self.trans1 = TransitionLayer(channels, int(channels * theta))
        channels = int(channels * theta)

        self.denseblock2 = DenseBlock(block_config[1], channels, growth_rate)
        channels += block_config[1] * growth_rate

        self.trans2 = TransitionLayer(channels, int(channels * theta))
        channels = int(channels * theta)

        self.denseblock3 = DenseBlock(block_config[2], channels, growth_rate)
        channels += block_config[2] * growth_rate

        self.trans3 = TransitionLayer(channels, int(channels * theta))
        channels = int(channels * theta)

        self.denseblock4 = DenseBlock(block_config[3], channels, growth_rate)
        channels += block_config[3] * growth_rate

        # 全局平均池化和分类器
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # [N, 3, 224, 224]
        x = self.conv1(x)
        # [N, 64, 112, 112]
        x = self.maxpool(x)
        # [N, 64, 112, 112]
        x = self.denseblock1(x)
        # [N, 256, 112, 112]
        x = self.trans1(x)
        # [N, 128, 56, 56]
        x = self.denseblock2(x)
        # [N, 496, 56, 56]
        x = self.trans2(x)
        # [N, 248, 28, 28]
        x = self.denseblock3(x)
        # [N, 1008, 28, 28]
        x = self.trans3(x)
        # [N, 504, 14, 14]
        x = self.denseblock4(x)
        # [N, 1016, 14, 14]

        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def densenet121(num_classes: int = 1000) -> DenseNet:
    return DenseNet(num_classes=num_classes, growth_rate=32, block_config=(6, 12, 24, 16))


def densenet169(num_classes: int = 1000) -> DenseNet:
    return DenseNet(num_classes=num_classes, growth_rate=32, block_config=(6, 12, 32, 32))


def densenet201(num_classes: int = 1000) -> DenseNet:
    return DenseNet(num_classes=num_classes, growth_rate=32, block_config=(6, 12, 48, 32))


def densenet264(num_classes: int = 1000) -> DenseNet:
    return DenseNet(num_classes=num_classes, growth_rate=32, block_config=(6, 12, 64, 48))


def densenet161(num_classes: int = 1000) -> DenseNet:
    return DenseNet(num_classes=num_classes, growth_rate=48, block_config=(6, 12, 36, 24))
