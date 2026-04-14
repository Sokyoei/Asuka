"""
ShuffleNet

SeeAlso:
- [torchvision.models.shufflenetv2](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)
"""

import math

import torch
from torch import Tensor, nn

from Ahri.Asuka.nn import Conv, DWConv


def make_divisible(v, divisor=8):
    return int(math.ceil(v / divisor) * divisor)


class ChannelShuffle(nn.Module):
    """通道混洗"""

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        # reshape [N, C, H, W] -> [N, G, C/G, H, W]
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        # transpose [N, G, C/G, H, W] -> [N, C/G, G, H, W]
        x = x.transpose(1, 2).contiguous()
        # flatten [N, C/G, G, H, W] -> [N, C, H, W]
        x = x.view(batch_size, -1, height, width)
        return x


class ShuffleUnitV1(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, groups: int = 3):
        super().__init__()
        if stride == 1:
            assert in_channels == out_channels, "stride=1 时输入输出通道数必须相同"

        self.stride = stride
        self.groups = groups

        self.bottleneck_channels = make_divisible(out_channels // 4, groups)

        self.blocks = nn.Sequential(
            Conv(in_channels, self.bottleneck_channels, kernel_size=1, groups=groups, bias=False),
            ChannelShuffle(groups),
            DWConv(self.bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False, activation=None),
            Conv(
                self.bottleneck_channels,
                out_channels if stride == 1 else out_channels - in_channels,
                kernel_size=1,
                groups=groups,
                bias=False,
                activation=None,
            ),
        )
        self.shortcut = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if self.stride == 2 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.blocks(x)
        residual = self.shortcut(x)

        out = out + residual if self.stride == 1 else torch.cat([out, residual], dim=1)
        out = self.relu(out)
        return out


class ShuffleNetV1(nn.Module):

    def __init__(self, num_classes: int = 1000, groups: int = 3):
        super().__init__()
        self.groups = groups

        self.conv1 = Conv(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(24, 240, 4, 2)
        self.stage3 = self._make_stage(240, 480, 8, 2)
        self.stage4 = self._make_stage(480, 960, 4, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(960, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # 第一个 block，stride=2，需要调整通道数
        layers.append(ShuffleUnitV1(in_channels, out_channels, stride=stride, groups=self.groups))
        # 后续 blocks，stride=1
        for _ in range(num_blocks - 1):
            layers.append(ShuffleUnitV1(out_channels, out_channels, stride=1, groups=self.groups))

        return nn.Sequential(*layers)

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
        # [N, 24, 112, 112]
        x = self.maxpool(x)
        # [N, 24, 56, 56]
        x = self.stage2(x)
        # [N, 240, 28, 28]
        x = self.stage3(x)
        # [N, 480, 14, 14]
        x = self.stage4(x)
        # [N, 960, 7, 7]
        x = self.global_pool(x)
        # [N, 960, 1, 1]
        x = torch.flatten(x, 1)
        # [N, 960]
        x = self.fc(x)
        # [N, num_classes]
        return x


def shufflenet_v1(num_classes: int = 1000, groups: int = 3) -> ShuffleNetV1:
    return ShuffleNetV1(num_classes=num_classes, groups=groups)
