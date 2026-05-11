import torch


def main():
    x = torch.range(1, 6, dtype=torch.int).reshape(2, 3)
    y = torch.range(6, 1, -1, dtype=torch.int).reshape(3, 2)
    print(x)
    print(y)
    print(torch.matmul(x, y))


if __name__ == "__main__":
    main()
