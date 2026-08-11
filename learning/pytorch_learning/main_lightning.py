"""
使用 [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) 进行模型训练
"""

import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from Ahri.Asuka.config.config import settings

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整图像大小以适应 ResNet-18 的输入尺寸
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道图像转换为三通道
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ]
)


class Model(L.LightningModule):

    def __init__(self, lr: float = 0.001):
        super().__init__()
        self.lr = lr

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # MNIST 是 10 分类
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor):
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class Data(L.LightningDataModule):

    def prepare_data(self) -> None:
        torchvision.datasets.MNIST(root=settings.DATA_DIR, train=True, download=True)
        torchvision.datasets.MNIST(root=settings.DATA_DIR, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.trainset = torchvision.datasets.MNIST(
                root=settings.DATA_DIR, train=True, download=False, transform=transform
            )
            self.valset = torchvision.datasets.MNIST(
                root=settings.DATA_DIR, train=False, download=False, transform=transform
            )
        if stage == "test":
            self.testset = torchvision.datasets.MNIST(
                root=settings.DATA_DIR, train=False, download=False, transform=transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=64, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valset, batch_size=64, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.testset, batch_size=64, shuffle=False, num_workers=0)


def main():
    # 训练
    model = Model()
    data = Data()
    tb_logger = TensorBoardLogger(save_dir=settings.LOG_DIR, name="lightning_mnist")
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    ckpt = ModelCheckpoint(
        dirpath=settings.MODELS_DIR,
        filename="mnist-{epoch:02d}-{val_loss:.5f}",
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        save_last=True,  # 保存最优模型时，额外复制一份，可以断点续训，Windows/Linux 兼容
        save_on_exception=True,  # 训练报错、崩溃时自动保存当前权重
        save_weights_only=False,  # 保存完整模型，包含模型参数
        save_on_train_epoch_end=False,  # 验证集跑完再判断保存模型
    )
    trainer = L.Trainer(
        max_epochs=10,  # 训练批次数
        accelerator="auto",  # 自动检测加速设备，CUDA, NPU 等
        logger=tb_logger,
        enable_checkpointing=True,  # 自动保存最优模型
        callbacks=[early_stop, ckpt],
    )
    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    # 验证
    if ckpt.best_model_path is not None:
        best_model = Model.load_from_checkpoint(ckpt.best_model_path)
    else:
        raise FileNotFoundError("未保存任何最优模型，请检查参数设置")
    best_model.eval()
    test_loader = data.test_dataloader()
    x, y = next(iter(test_loader))
    with torch.no_grad():
        pred = torch.argmax(best_model(x), dim=1)
    print(f"真实标签: {y[0].item()}, 预测标签: {pred[0].item()}")


if __name__ == '__main__':
    main()
