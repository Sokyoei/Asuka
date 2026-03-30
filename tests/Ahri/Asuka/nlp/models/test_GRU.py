import pytest
import torch

from Ahri.Asuka.nlp.models import GRU
from Ahri.Asuka.utils import DEVICE


def test_GRU():
    input_size = 10
    hidden_size = 20
    num_layers = 2
    output_size = 1
    batch_size = 3
    seq_len = 5

    model = GRU(input_size, hidden_size, num_layers, output_size).to(DEVICE)
    model.eval()

    x = torch.randn(batch_size, seq_len, input_size, device=DEVICE)
    h0 = model.init_hidden(batch_size, DEVICE)
    output, hn = model(x, h0)

    assert output.shape == (batch_size, output_size)
    assert hn.shape == (num_layers, batch_size, hidden_size)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
