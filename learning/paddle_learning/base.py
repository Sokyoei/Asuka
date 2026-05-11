import paddle


def main():
    print(f"support cuda: {paddle.device.is_compiled_with_cuda()}")


if __name__ == "__main__":
    main()
