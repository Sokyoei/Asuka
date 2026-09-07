import pytest
import torch

from asuka.utils import DEVICE
from asuka.vision.models import densenet121, densenet161, densenet169, densenet201, densenet264


def test_DenseNet():
    for densenet in [densenet121, densenet161, densenet169, densenet201, densenet264]:
        model = densenet().to(DEVICE)
        model.eval()

        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        y: torch.Tensor = model(x)
        assert y.shape == (1, 1000)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
