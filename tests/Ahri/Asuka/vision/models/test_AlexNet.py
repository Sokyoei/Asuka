import pytest
import torch

from Ahri.Asuka.utils import DEVICE
from Ahri.Asuka.vision.models import alexnet


def test_AlexNet():
    model = alexnet().to(DEVICE)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=DEVICE)
    y: torch.Tensor = model(x)
    assert y.shape == (1, 1000)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
