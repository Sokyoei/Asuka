from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


def one_hot_encode(data: NDArray | torch.Tensor, max_len: int):
    if isinstance(data, NDArray):
        return np.eye(max_len, dtype=np.float32)[data]
    elif isinstance(data, torch.Tensor):
        return F.one_hot(data, max_len)
    else:
        raise NotImplementedError(f"{type(data)} are not support.")
