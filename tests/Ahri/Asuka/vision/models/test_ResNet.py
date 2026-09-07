import pytest
import torch

from asuka.utils import DEVICE
from asuka.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def test_ResNet():
    for resnet in [resnet18, resnet34, resnet50, resnet101, resnet152]:
        model = resnet().to(DEVICE)
        model.eval()

        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        y: torch.Tensor = model(x)
        assert y.shape == (1, 1000)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
