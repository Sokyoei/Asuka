import torch
from torch import Tensor, nn


class RNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层神经元个数
        self.num_layers = num_layers  # RNN 堆叠层数

        # batch_first=True, 输入形状变为 [N, seq_len, feature_dim]
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,  # 仅多层时启用 Dropout
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        # x: [N, seq_len, input_size]
        # hidden: [num_layers, N, hidden_size]
        out, hidden = self.rnn(x, hidden)
        # out: 所有时间步的隐藏状态 [N, seq_len, hidden_size]
        # hidden: 最后一个时间步的隐藏状态 [num_layers, N, hidden_size]
        out = out[:, -1, :]
        # [N, hidden_size]
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size: int, device: torch.device = "cpu") -> Tensor:
        # [num_layers, N, hidden_size]
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
