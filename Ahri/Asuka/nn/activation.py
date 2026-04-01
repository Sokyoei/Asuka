import torch.nn.functional as F

ACTIVATIONS = {
    # has inplace
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "selu": F.selu,
    "elu": F.elu,
    "celu": F.celu,
    "mish": F.mish,
    "silu": F.silu,
    "hardtanh": F.hardtanh,
    # no inplace args
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "softplus": F.softplus,
    "tanhshrink": F.tanhshrink,
    "softsign": F.softsign,
    "softmin": F.softmin,
    "glu": F.glu,
    "tanh": F.tanh,
}
