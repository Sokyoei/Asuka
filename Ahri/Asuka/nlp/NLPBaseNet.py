from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
EPOCHS = 50
LR = 0.01

text = ["how are you?", "I am fine, thank you, and you?", "I am fine too."]
chars = set(''.join(text))
int2char = dict(enumerate(chars))
char2int = {char: i for i, char in int2char.items()}

max_len = len(max(text, key=len))
# padding
for t in range(len(text)):
    while len(text[t]) < max_len:
        text[t] += ' '


input_seq = []
target_seq = []
for i in range(len(text)):
    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])


for i in range(len(text)):
    input_seq[i] = [char2int[c] for c in input_seq[i]]
    target_seq[i] = [char2int[c] for c in target_seq[i]]

dict_size = len(char2int)
seq_len = max_len - 1
batch_size = len(text)


def one_hot_encode(data, max_len: int):
    return np.eye(dict_size, dtype=np.float32)[data]


input_seq = [one_hot_encode(seq, max_len) for seq in input_seq]

input_seq = torch.from_numpy(np.array(input_seq))
target_seq = torch.Tensor(target_seq)


class NLPBaseNet(nn.Module):

    def __init__(
        self,
        kernel: Literal["RNN", "LSTM", "GRU"],
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if kernel == "RNN":
            self.kernel = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif kernel == "LSTM":
            self.kernel = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif kernel == "GRU":
            self.kernel = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise AttributeError(f"{kernel} 不支持。")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, hidden = self.kernel(x, hidden)
        out: Tensor
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden


def train():
    net = NLPBaseNet("RNN", dict_size, dict_size, 100, 1)
    net.to(DEVICE)
    loss = nn.CrossEntropyLoss()
    loss.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), LR)

    for i in range(EPOCHS):
        optimizer.zero_grad()
        input_seq.to(DEVICE)
        y_pred, _hidden_pred = net(input_seq)
        loss_value: Tensor = loss(y_pred, target_seq.view(-1).long())
        loss_value.backward()
        optimizer.step()
        print(f"epoch: {i + 1:>3}, loss: {loss_value}")
    return net


def predict(model, character):
    character = np.array([[char2int[c] for c in character]])
    # character = one_hot_encode(character, dict_size, character.shape[1], 1)
    # character = torch.from_numpy(character)
    character = [one_hot_encode(seq, max_len) for seq in character]

    character = torch.from_numpy(np.array(character))
    out, hidden = model(character)
    prob = nn.functional.softmax(out[-1], dim=0).data
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden


def sample(model: nn.Module, out_len, start='hey'):
    model.eval()  # eval mode
    start = start.lower()
    chars = [ch for ch in start]
    size = out_len - len(chars)
    for ii in range(size):
        char, _h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)


def main():
    net = train()
    print(sample(net, 15, "h"))


if __name__ == "__main__":
    main()
