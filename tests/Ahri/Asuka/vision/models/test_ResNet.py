import pytest
import torch

from Ahri.Asuka.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def test_ResNet():
    for resnet in [resnet18, resnet34, resnet50, resnet101, resnet152]:
        model = resnet()
        x = torch.randn(1, 3, 224, 224)
        y: torch.Tensor = model(x)
        assert y.shape == (1, 1000)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
