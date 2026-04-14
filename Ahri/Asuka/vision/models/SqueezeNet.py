"""
SqueezeNet

SeeAlso:
- [torchvision.models.squeezenet](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)
"""

from typing import Literal

import torch
from torch import Tensor, nn


class Fire(nn.Module):

    def __init__(self, in_channels: int, squeeze_channels: int, expand1x1_channels: int, expand3x3_channels: int):
        super(Fire, self).__init__()

        # Squeeze layer
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        # Expand layer
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)

        expand1x1_out = self.expand1x1_activation(self.expand1x1(x))
        expand3x3_out = self.expand3x3_activation(self.expand3x3(x))
        return torch.cat([expand1x1_out, expand3x3_out], 1)


class SqueezeNet(nn.Module):

    def __init__(self, num_classes: int = 1000, p: float = 0.5, version: Literal["1_0", "1_1"] = "1_1"):
        super(SqueezeNet, self).__init__()
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Invalid version, please choose from '1_0' or '1_1'")

        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

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
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def squeezenet_v1_0(num_classes: int = 1000) -> SqueezeNet:
    return SqueezeNet(num_classes, version="1_0")


def squeezenet_v1_1(num_classes: int = 1000) -> SqueezeNet:
    return SqueezeNet(num_classes, version="1_1")
