import pytest
import torch

from Ahri.Asuka.utils import DEVICE
from Ahri.Asuka.vision.models import googlenet


def test_GoogLeNet():
    model = googlenet().to(DEVICE)
    model.train()

    x = torch.randn(1, 3, 224, 224, device=DEVICE)
    y, aux1, aux2 = model(x)
    y: torch.Tensor
    aux1: torch.Tensor
    aux2: torch.Tensor
    assert y.shape == (1, 1000)
    assert aux1.shape == (1, 1000)
    assert aux2.shape == (1, 1000)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
