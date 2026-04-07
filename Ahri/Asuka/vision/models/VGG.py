from typing import Literal

import torch
from torch import Tensor, nn


class VGG(nn.Module):

    def __init__(self, num_classes: int = 1000, vgg_type: Literal["vgg11", "vgg13", "vgg16", "vgg19"] = "vgg16"):
        super(VGG, self).__init__()
        # fmt:off
        cfg: dict[str, list[int | str]] = {
            #         block1       block2         block3                   block4                   block5
            "vgg11": [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
            "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
            "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
            "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        # fmt:on

        self.features = self._make_layers(cfg[vgg_type])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg11(num_classes=1000):
    return VGG(num_classes=num_classes, vgg_type="vgg11")


def vgg13(num_classes=1000):
    return VGG(num_classes=num_classes, vgg_type="vgg13")


def vgg16(num_classes=1000):
    return VGG(num_classes=num_classes, vgg_type="vgg16")


def vgg19(num_classes=1000):
    return VGG(num_classes=num_classes, vgg_type="vgg19")
