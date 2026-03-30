import torch
from torch import Tensor, nn


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,  # 仅多层时启用 Dropout
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor]):
        out, (h, c) = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device = "cpu"):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0
