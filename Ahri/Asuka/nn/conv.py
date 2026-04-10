from typing import Literal

from torch import Tensor, nn

from .activation import ACTIVATIONS


class Conv(nn.Module):
    """
    Standard Convolution「标准卷积」: Conv2d + BatchNorm2d + Activation

    Notes:
        params = kernel_size * kernel_size * in_channels * out_channels + bias(optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: Literal["same", "valid"] | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        activation: str = "relu",
        is_in: bool = False,
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 步长、步幅
            padding=padding,  # 填充
            dilation=dilation,  # 空洞卷积
            groups=groups,  # 分组卷积
            bias=bias,  # 是否使用偏置
            padding_mode=padding_mode,  # 填充模式
            device=device,
            dtype=dtype,
        )
        if is_in:
            self.bn = nn.InstanceNorm2d(num_features=out_channels)
        else:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class DWConv(Conv):
    """
    Depthwise Convolution「深度卷积」

    Notes:
        params = kernel_size * kernel_size * in_channels + bias(optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding=0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
        padding_mode="zeros",
        device=None,
        dtype=None,
        activation: str = "relu",
    ):
        assert in_channels == out_channels, "in_channels should be equal to out_channels"
        super(DWConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            activation=activation,
        )


class PWConv(Conv):
    """
    Pointwise Convolution「逐点卷积」

    Notes:
        params = 1 * 1 * in_channels * out_channels + bias(optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
        padding=0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode="zeros",
        device=None,
        dtype=None,
        activation: str = "relu",
    ):
        super(PWConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            activation=activation,
        )


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution「深度可分离卷积」

    Examples:
        >>> import torch
        >>> x = torch.randn(2, 3, 32, 32)
        >>> model = DSConv(3, 26, 3, 1, 1, activation="relu")
        >>> out = model(x)
        >>> out.shape
        torch.Size([2, 26, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding=0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
        padding_mode="zeros",
        device=None,
        dtype=None,
        activation: str = "relu",
    ):
        super(DSConv, self).__init__()
        self.dw = DWConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            activation=activation,
        )
        self.pw = PWConv(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            device=device,
            dtype=dtype,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x
