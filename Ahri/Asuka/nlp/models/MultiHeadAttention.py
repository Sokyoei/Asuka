"""
多头注意力机制
"""

from torch import Tensor, nn


class MultiHeadAttention(nn.Module):

    def __init__(self, dim: int, n_head: int, dk: int, dv: int):
        super().__init__()
        self.n_head = n_head  # 注意力头个数
        self.dk = dk
        self.dv = dv

        self.scale = dk**-0.5  # 缩放因子

        self.q = nn.Linear(dim, n_head * dk)
        self.k = nn.Linear(dim, n_head * dk)
        self.v = nn.Linear(dim, n_head * dv)

        self.out_proj = nn.Linear(n_head * dv, dim)

    def forward(self, x: Tensor):
        batch_size, seq_len, dim = x.shape  # noqa: RUF059

        q: Tensor = self.q(x)
        k: Tensor = self.k(x)
        v: Tensor = self.v(x)

        # [N x seq_len x dim] -> [N x seq_len x n_head x dk] -> [N x n_head x seq_len x dk]
        q = q.view(batch_size, seq_len, self.n_head, self.dk).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.dk).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.dv).transpose(1, 2)

        attn: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        # [N x n_head x seq_len x dv] -> [N x seq_len x n_head*dv]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.dv)

        out = self.out_proj(out)
        return out
