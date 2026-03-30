import pytest
import torch

from Ahri.Asuka.nlp.models import RNN
from Ahri.Asuka.utils import DEVICE


def test_RNN():
    # 超参数
    input_size = 10  # 每一步的特征数
    hidden_size = 20  # 隐藏层神经元个数
    num_layers = 3  # RNN 层数
    output_size = 5  # 分类类别个数
    batch_size = 1  # 批次数
    seq_len = 3  # 序列长度

    model = RNN(input_size, hidden_size, num_layers, output_size).to(DEVICE)
    model.eval()

    x = torch.randn(batch_size, seq_len, input_size, device=DEVICE)
    h0 = model.init_hidden(batch_size, DEVICE)
    output, hn = model(x, h0)

    assert output.shape == (batch_size, output_size)
    assert hn.shape == (num_layers, batch_size, hidden_size)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
