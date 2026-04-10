from typing import Literal

from torch import nn


def init_weights(
    net: nn.Module,
    conv2d_mode: Literal["fan_in", "fan_out"] = "fan_in",
    conv2d_nonlinearity: Literal["leaky_relu", "relu"] = "leaky_relu",
):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode=conv2d_mode, nonlinearity=conv2d_nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
