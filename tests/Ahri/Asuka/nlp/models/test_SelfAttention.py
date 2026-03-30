import pytest
import torch

from Ahri.Asuka.nlp.models import SelfAttention


def test_SelfAttention():
    attn = SelfAttention(2, 2, 3)
    x = torch.randn((1, 4, 2))
    out: torch.Tensor = attn(x)
    assert out.shape == torch.Size([1, 4, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
