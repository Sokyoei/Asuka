import pytest
import torch

from Ahri.Asuka.models.MultiHeadAttention import MultiHeadAttention


def test_MultiHeadAttention():
    dim = 16
    n_head = 2
    dk = 8
    dv = 8

    model = MultiHeadAttention(dim, n_head, dk, dv)
    x = torch.randn(1, 5, dim)
    output: torch.Tensor = model(x)

    assert output.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__])
