import mindspore as ms
import numpy as np
from mindspore import ops

ms.set_context(device_target="CPU")


def main():
    x = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
    y = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
    result = ops.add(x, y)
    print(result)


if __name__ == "__main__":
    main()
