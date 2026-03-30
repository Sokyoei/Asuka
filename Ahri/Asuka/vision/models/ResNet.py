# ruff: noqa: F401
import torch
from torch import nn
from torchvision import datasets, models
from torchvision.datasets import FashionMNIST

EPOCHS = 10


def main():
    net = models.resnet50()  # noqa: F841


if __name__ == "__main__":
    main()
