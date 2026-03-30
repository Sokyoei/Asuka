"""
Self-Attention 注意力机制
"""

from torch import Tensor, nn


class SelfAttention(nn.Module):
    r"""
    $$
        Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
    $$
    """

    def __init__(self, dim: int, dk: int, dv: int) -> None:
        super().__init__()
        self.scale = dk**-0.5  # 缩放因子
        self.q = nn.Linear(dim, dk)  # queries 查询
        self.k = nn.Linear(dim, dk)  # keys    关键词
        self.v = nn.Linear(dim, dv)  # values  值

    def forward(self, x: Tensor):
        q: Tensor = self.q(x)
        k: Tensor = self.k(x)
        v: Tensor = self.v(x)

        attn: Tensor = (q @ k.transpose(-2, -1)) * self.scale  # 计算相似度
        attn = attn.softmax(dim=-1)  # 归一化
        out = attn @ v  # 加权求和
        return out
