import pytest
import torch

from asuka.utils import DEVICE
from asuka.vision.models import lenet5


def test_LeNet5():
    model = lenet5().to(DEVICE)
    model.eval()

    x = torch.randn(1, 1, 32, 32, device=DEVICE)
    y: torch.Tensor = model(x)
    assert y.shape == (1, 1000)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
