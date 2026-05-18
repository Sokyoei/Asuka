import functools

import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.onnx import register_custom_op_symbolic
from torch.onnx._internal import jit_utils, registration

MODEL_ASINH_NAME = "asinh.onnx"
REGISTER_METHOD = 1


if REGISTER_METHOD == 1:
    _onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=9)

    @_onnx_symbolic("aten::asinh")
    def asinh_symbolic(g: jit_utils.GraphContext, input, *, out=None):
        return g.op("Asinh", input)

else:

    def asinh_symbolic(g: jit_utils.GraphContext, input, *, out=None):
        return g.op("Asinh", input)

    register_custom_op_symbolic("aten::asinh", asinh_symbolic, 12)


class AsinhModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


model = AsinhModel()
x: Tensor = torch.randn(1, 5)


def inference():
    model.eval()
    y: Tensor = model(x)
    print(f"{x.data=}")
    print(f"torch: {y.data=}")


def export():
    torch.onnx.export(
        model,
        (x,),
        MODEL_ASINH_NAME,
        input_names=["input"],
        output_names=["output"],
        # ## ERROR
        # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::asinh' to ONNX opset version 9 is
        # not supported. Please feel free to request support or submit a pull request on PyTorch GitHub:
        # https://github.com/pytorch/pytorch/issues.
        #
        # 但是 ONNX opset9 的官方文档是支持的(https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asinh)，所以 onnx
        # 和 pytorch 之间没有建立 asinh 的映射。
        #
        opset_version=12,
    )


def onnx_val():
    model.eval()
    sess = ort.InferenceSession(MODEL_ASINH_NAME)
    y = sess.run(None, {"input": x.numpy()})
    print(f"onnxruntime: {y=}")


def main():
    inference()
    export()
    onnx_val()


if __name__ == "__main__":
    main()
