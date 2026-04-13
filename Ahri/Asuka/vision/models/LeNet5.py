"""
LeNet-5 1998
"""

from torch import Tensor, nn


class LeNet5(nn.Module):

    def __init__(self, n_classes: int = 1000) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5, 1), nn.BatchNorm2d(6), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5, 1), nn.BatchNorm2d(16), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # [N, 1, 32, 32]
        x1 = self.conv1(x)
        # [N, 6, 28, 28]
        x2 = self.pool(x1)
        # [N, 6, 14, 14]
        x3 = self.conv2(x2)
        # [N, 16, 10, 10]
        x4 = self.pool(x3)
        # [N, 16, 5, 5]
        x4: Tensor = x4.reshape(x4.size(0), -1)
        # [N, 400]
        x5 = self.fc1(x4)
        # [N, 120]
        x6 = self.fc2(x5)
        # [N, 84]
        x7 = self.fc3(x6)
        # [N, n_classes]
        return x7


def lenet5(num_classes: int = 1000):
    return LeNet5(num_classes)
