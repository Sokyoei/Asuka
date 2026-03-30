import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Ahri.Asuka.config.config import settings
from Ahri.Asuka.utils import DEVICE
from Ahri.Asuka.vision.models import LeNet5

EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 1


def mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    train_images = datasets.MNIST(settings.DATA_DIR, True, transform, download=True)
    test_images = datasets.MNIST(settings.DATA_DIR, False, transform, download=True)
    train_data = DataLoader(train_images, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    test_data = DataLoader(test_images, BATCH_SIZE, num_workers=NUM_WORKERS)
    return train_data, test_data


train_data, test_data = mnist()


def train():
    net = LeNet5(10).to(DEVICE)
    losser = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)

    for i in range(EPOCHS):
        sum_loss = 0
        for X_train, y_train in train_data:
            X_train: Tensor
            y_train: Tensor
            X_train = X_train.to(DEVICE)
            y_train = y_train.to(DEVICE)
            optimizer.zero_grad()
            y_pred = net(X_train)
            loss: Tensor = losser(y_pred, y_train)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        # test
        total = 0
        correct = 0
        for X_test, y_test in test_data:
            X_test: Tensor
            y_test: Tensor
            X_test = X_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            y_pred: Tensor = net(X_test)
            _, predict = torch.max(y_pred.data, 1)
            total += y_test.size(0)
            correct += (predict == y_test).sum()

        print(f"epoch: {i + 1}, train_loss: {sum_loss / len(train_data)}, test_acc: {correct / total}")

    torch.save(net.state_dict(), settings.DATA_DIR / "LeNet5.pt")


def main():
    train()


if __name__ == "__main__":
    main()
