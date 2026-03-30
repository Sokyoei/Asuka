import pytest
import torch

from Ahri.Asuka.nlp.models import LSTM
from Ahri.Asuka.utils import DEVICE


def test_LSTM():
    input_size = 10  # 每步特征数
    hidden_size = 20  # 隐藏层维度
    num_layers = 2  # 层数
    output_size = 1  # 输出维度（回归任务）
    batch_size = 3  # 批次大小
    seq_len = 5  # 序列长度

    model = LSTM(input_size, hidden_size, num_layers, output_size, dropout=0.5).to(DEVICE)
    model.eval()

    x = torch.randn(batch_size, seq_len, input_size, device=DEVICE)
    h0, c0 = model.init_hidden(batch_size, DEVICE)
    out, (hn, cn) = model(x, (h0, c0))

    assert out.shape == (batch_size, output_size)
    assert hn.shape == (num_layers, batch_size, hidden_size)
    assert cn.shape == (num_layers, batch_size, hidden_size)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
