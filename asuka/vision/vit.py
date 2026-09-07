"""
Vision Transformer
"""

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from asuka import ASUKA_ROOT
from asuka.utils import DEVICE


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels, patch_size, embed_dim, num_patchs, dropout) -> None:
        super().__init__()
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, embed_dim, patch_size, patch_size), nn.Flatten(2))
        self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim), requires_grad=True))
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patchs + 1, embed_dim), requires_grad=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class Vit(nn.Module):
    def __init__(
        self, in_channels, patch_size, embed_dim, num_patchs, dropout, num_heads, activation, num_encoders, num_classes
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patchs, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_blocks(x)
        x = self.mlp(x[:, 0, :])
        return x


def torch_vit():
    test = datasets.CIFAR10(
        ASUKA_ROOT / "models", False, transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]), download=True
    )
    test_data = DataLoader(test, 64)

    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(DEVICE)
    vit.eval()

    with torch.no_grad():  # 关闭梯度计算，节省显存并提升速度
        total_correct = 0
        total_samples = 0

        for x_test, y_test in test_data:
            x_test = x_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            y_pred: Tensor = vit(x_test)

            pred_labels = torch.argmax(y_pred, dim=1)
            batch_correct = (pred_labels == y_test).sum().item()
            batch_samples = x_test.shape[0]

            total_correct += batch_correct
            total_samples += batch_samples

        accuracy = total_correct / total_samples
        logger.info(f"total: {total_samples}, correct: {total_correct}, accuracy: {accuracy:.4f}")


def main():
    torch_vit()
    vit = Vit(224, 16, 128, 20, True, 1, F.relu, 4, 10).to(DEVICE)
    x = torch.randn(1, 3, 224, 224).to(DEVICE)
    out: Tensor = vit(x)
    print(out.shape)


if __name__ == "__main__":
    main()
