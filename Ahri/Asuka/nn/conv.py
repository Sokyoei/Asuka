from typing import Literal

from torch import Tensor, nn


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
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        activation: type[nn.Module] | None = nn.ReLU,
        inplace: bool | None = True,
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
        if norm is not None:
            self.norm = norm(num_features=out_channels, device=device, dtype=dtype)
        if activation is None:
            activation = nn.Identity
        # 检查激活函数是否支持 inplace 参数
        params = {}
        if inplace and "inplace" in activation.__init__.__code__.co_varnames:
            params["inplace"] = inplace
        self.activation = activation(**params)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        x = self.activation(x)
        return x


class DWConv(Conv):
    """
    Depthwise Convolution「深度卷积」

    Notes:
        params = kernel_size * kernel_size * in_channels + bias(optional)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding=0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
        padding_mode="zeros",
        device=None,
        dtype=None,
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        activation: type[nn.Module] | None = nn.ReLU,
        inplace: bool | None = True,
    ):
        super(DWConv, self).__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            norm=norm,
            activation=activation,
            inplace=inplace,
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
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        activation: type[nn.Module] | None = nn.ReLU,
        inplace: bool | None = True,
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
            norm=norm,
            activation=activation,
            inplace=inplace,
        )


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution「深度可分离卷积」

    Examples:
        >>> import torch
        >>> x = torch.randn(2, 3, 32, 32)
        >>> model = DSConv(3, 26, 3, 1, 1)
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
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        activation: type[nn.Module] | None = nn.ReLU,
        inplace: bool | None = True,
    ):
        super(DSConv, self).__init__()
        self.dw = DWConv(
            channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            norm=norm,
            activation=activation,
            inplace=inplace,
        )
        self.pw = PWConv(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            device=device,
            dtype=dtype,
            norm=norm,
            activation=activation,
            inplace=inplace,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x
