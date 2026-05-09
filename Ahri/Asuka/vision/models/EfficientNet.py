"""
EfficientNet

SeeAlso:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [torchvision.models.efficientnet](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)
"""

import math
from typing import Literal

import torch
from torch import Tensor, nn

from Ahri.Asuka.nn import Conv
from Ahri.Asuka.nn.conv import DWConv


def stochastic_depth(input: Tensor, p: float, mode: Literal["batch", "row"], training: bool = True):
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":  # noqa: SIM108
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("stochastic_depth")


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: Literal["batch", "row"]) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)


def _make_divisible(v: float, divisor: int, min_value: int | None = None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation"""

    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced_channels = _make_divisible(in_channels * se_ratio, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class MBConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        se_ratio: float,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            Conv(in_channels, expanded_channels, 1, activation=nn.SiLU) if expand_ratio != 1 else nn.Identity(),
            DWConv(expanded_channels, kernel_size, stride, kernel_size // 2, activation=nn.SiLU),
            SqueezeExcitation(expanded_channels, se_ratio) if se_ratio > 0 else nn.Identity(),
            Conv(expanded_channels, out_channels, 1, activation=None),
        )
        self.stochastic_depth = StochasticDepth(drop_connect_rate, 'row')

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.use_res_connect:
            out = self.stochastic_depth(out)
            out += x
        return out


class EfficientNet(nn.Module):

    def __init__(
        self,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        num_classes: int = 1000,
        drop_connect_rate: float = 0.2,
    ):
        super(EfficientNet, self).__init__()

        # EfficientNet-B0 配置
        # fmt:off
        cfgs = [
            # expand_ratio, channels, repeats, stride, kernel_size, se_ratio
            [1,             16,       1,       1,      3,           0.25],  # stage2
            [6,             24,       2,       2,      3,           0.25],  # stage3
            [6,             40,       2,       2,      5,           0.25],  # stage4
            [6,             80,       3,       2,      3,           0.25],  # stage5
            [6,             112,      3,       1,      5,           0.25],  # stage6
            [6,             192,      4,       2,      5,           0.25],  # stage7
            [6,             320,      1,       1,      3,           0.25],  # stage8
        ]
        # fmt:on

        def adjust_channels(channels: int) -> int:
            return _make_divisible(channels * width_mult, 8)

        def adjust_depth(depth: int) -> int:
            return math.ceil(depth * depth_mult)

        stem_channels = adjust_channels(32)
        # stage1
        self.stem = Conv(3, stem_channels, 3, stride=2, padding=1, activation=nn.SiLU)

        # 构建 MBConv 层
        total_stage_blocks = sum(adjust_depth(cfg[2]) for cfg in cfgs)
        stage_block_id = 0
        layers = []
        in_channels = stem_channels

        for expand_ratio, channels, repeats, stride, kernel_size, se_ratio in cfgs:
            out_channels = adjust_channels(channels)
            repeats = adjust_depth(repeats)
            for i in range(repeats):
                # 除了第一层外，其余 stride=1
                current_stride = stride if i == 0 else 1
                layers.append(
                    MBConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=current_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        drop_connect_rate=drop_connect_rate * (stage_block_id / total_stage_blocks),
                    )
                )
                in_channels = out_channels
                stage_block_id += 1

        self.blocks = nn.Sequential(*layers)

        head_channels = adjust_channels(1280)
        # stage9
        self.head = Conv(in_channels, head_channels, 1, activation=nn.SiLU)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(head_channels, num_classes)

        # 权重初始化
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
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def efficientnet_b0(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=num_classes)


def efficientnet_b1(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.0, depth_mult=1.1, dropout_rate=0.2, num_classes=num_classes)


def efficientnet_b2(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.1, depth_mult=1.2, dropout_rate=0.3, num_classes=num_classes)


def efficientnet_b3(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.2, depth_mult=1.4, dropout_rate=0.3, num_classes=num_classes)


def efficientnet_b4(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.4, depth_mult=1.8, dropout_rate=0.4, num_classes=num_classes)


def efficientnet_b5(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.6, depth_mult=2.2, dropout_rate=0.4, num_classes=num_classes)


def efficientnet_b6(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=1.8, depth_mult=2.6, dropout_rate=0.5, num_classes=num_classes)


def efficientnet_b7(num_classes: int = 1000) -> EfficientNet:
    return EfficientNet(width_mult=2.0, depth_mult=3.1, dropout_rate=0.5, num_classes=num_classes)
