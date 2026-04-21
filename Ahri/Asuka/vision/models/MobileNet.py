"""
MobileNet

SeeAlso:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [torchvision.models.mobilenet](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenet.py)
"""

import torch
from torch import nn

from Ahri.Asuka.nn import DSConv


class MobileNetV1(nn.Module):

    def __init__(self, num_classes: int = 1000):
        super(MobileNetV1, self).__init__()

        config = [
            # out_channels, stride
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 深度可分离卷积层序列
        layers = []
        in_channels = 32
        for out_channels, stride in config:
            layers.append(DSConv(in_channels, out_channels, 3, stride, 1, activation=nn.ReLU6))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

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

    def forward(self, x):
        # [N, 3, 224, 224]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # [N, 32, 112, 112]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def mobilenet_v1(num_classes=1000):
    return MobileNetV1(num_classes)
