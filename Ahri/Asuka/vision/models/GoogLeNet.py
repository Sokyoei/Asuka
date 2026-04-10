import torch
from torch import Tensor, nn


class InceptionBlockOrigin(nn.Module):
    """论文中 Inception 原始结构"""

    def __init__(self, in_channels: int, ch1x1: int, ch3x3: int, ch5x5: int):
        super(InceptionBlockOrigin, self).__init__()

        # conv1x1
        self.branch1_conv1x1 = nn.Sequential(nn.Conv2d(in_channels, ch1x1, kernel_size=1), nn.ReLU(inplace=True))
        # conv3x3
        self.branch2_conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        # conv5x5
        self.branch3_conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5, kernel_size=5, padding=2), nn.ReLU(inplace=True)
        )
        # maxpool3x3
        self.branch4_maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1_conv1x1(x)
        x2 = self.branch2_conv3x3(x)
        x3 = self.branch3_conv5x5(x)
        x4 = self.branch4_maxpool3x3(x)
        # 将所有分支在通道维度上连接
        outputs = [x1, x2, x3, x4]
        return torch.cat(outputs, 1)


class InceptionBlock(nn.Module):

    def __init__(
        self, in_channels: int, ch1x1: int, ch3x3reduce: int, ch3x3: int, ch5x5reduce: int, ch5x5: int, pool_proj: int
    ):
        super(InceptionBlock, self).__init__()

        # conv1x1
        self.branch1_conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1), nn.BatchNorm2d(ch1x1), nn.ReLU(inplace=True)
        )
        # conv1x1 + conv3x3
        self.branch2_conv1x1_conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
        )
        # conv1x1 + conv5x5
        self.branch3_conv1x1_conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
        )
        # maxpool3x3 + conv1x1
        self.branch4_maxpool3x3_conv1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1_conv1x1(x)
        x2 = self.branch2_conv1x1_conv3x3(x)
        x3 = self.branch3_conv1x1_conv5x5(x)
        x4 = self.branch4_maxpool3x3_conv1x1(x)
        # 将所有分支在通道维度上连接
        outputs = [x1, x2, x3, x4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # 初始卷积层和池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 模块序列
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # 辅助分类器（如果启用）
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor] | Tensor:
        # [N, 3, 224, 224]
        x = self.conv1(x)
        # [N, 64, 112, 112]
        x = self.maxpool1(x)
        # [N, 64, 56, 56]
        x = self.conv2(x)
        # [N, 192, 56, 56]
        x = self.maxpool2(x)
        # [N, 192, 28, 28]

        x = self.inception3a(x)
        # [N, 256, 28, 28]
        x = self.inception3b(x)
        # [N, 480, 28, 28]
        x = self.maxpool3(x)
        # [N, 480, 14, 14]

        x = self.inception4a(x)
        # [N, 512, 14, 14]
        # 如果启用了辅助分类器，计算辅助损失
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # [N, 512, 14, 14]
        x = self.inception4c(x)
        # [N, 512, 14, 14]
        x = self.inception4d(x)
        # [N, 528, 14, 14]
        # 如果启用了辅助分类器，计算辅助损失
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # [N, 832, 14, 14]
        x = self.maxpool4(x)
        # [N, 832, 7, 7]

        x = self.inception5a(x)
        # [N, 832, 7, 7]
        x = self.inception5b(x)
        # [N, 1024, 7, 7]

        x = self.avgpool(x)
        # [N, 1024, 1, 1]
        x = torch.flatten(x, 1)
        # [N, 1024]
        x = self.dropout(x)
        x = self.fc(x)
        # [N, num_classes]

        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x


class InceptionAux(nn.Module):
    """辅助分类器"""

    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),  # 128 * 4 * 4 = 2048 when input is 14x14
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def googlenet(num_classes=1000, aux_logits=True):
    return GoogLeNet(num_classes=num_classes, aux_logits=aux_logits)
