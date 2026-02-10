import torch
import torchvision
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from Ahri.Asuka.config.config import settings


def main():
    # 定义数据转换
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整图像大小以适应ResNet-18的输入尺寸
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道图像转换为三通道
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
        ]
    )

    # 下载并加载训练集和测试集
    trainset = torchvision.datasets.MNIST(root=settings.DATA_DIR, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root=settings.DATA_DIR, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # 加载预训练的ResNet-18模型
    model = resnet18(pretrained=True)

    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    # train
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}")

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total}%")

    # save model
    torch.save(model, settings.MODELS_DIR / "resnet18_MNIST.pt")
    # torch.save(model.state_dict(), "resnet18_MNIST.pt")


if __name__ == "__main__":
    main()
