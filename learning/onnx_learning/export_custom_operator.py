import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from onnxruntime_extensions import PyCustomOpDef, onnx_op
from onnxruntime_extensions import get_library_path as _lib_path
from torch import Tensor, nn
from torch.autograd.function import Function, FunctionCtx
from torch.onnx._internal import jit_utils

MODEL_CUSTOM_OP_NAME = "custom_op.onnx"
SYMBOLIC = True


class AhriClip(Function):

    if SYMBOLIC:

        @staticmethod
        def symbolic(g: jit_utils.GraphContext, x: torch.Value) -> torch.Value:
            """注册算子，可以将散列的一些算子整合成一个算子"""
            x_shape = x.type().sizes()
            y_shape = x_shape
            y = g.op("Ahri::clip", x)
            y.setType(x.type().with_sizes(y_shape))
            return y

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        x = x.clamp(0)
        return x / (1 + torch.exp(-x))


ahri_clip = AhriClip.apply


class AhriClipModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ahri_clip(x)


model = AhriClipModel()
x: Tensor = torch.randn(1, 5)

if SYMBOLIC:

    @onnx_op(op_type="Ahri::clip", inputs=[PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float])
    def ahri_clip_ort(x: NDArray) -> NDArray:
        x = np.clip(x, 0, np.inf)
        return x / (1 + np.exp(-x))

    so = ort.SessionOptions()
    so.register_custom_ops_library(_lib_path())


def inference():
    model.eval()
    y: Tensor = model(x)
    print(f"{x.data=}")
    print(f"torch: {y.data=}")


def export():
    torch.onnx.export(
        model, (x,), MODEL_CUSTOM_OP_NAME, input_names=["input"], output_names=["output"], opset_version=12
    )


def onnx_val():
    model.eval()
    sess = ort.InferenceSession(MODEL_CUSTOM_OP_NAME, so if SYMBOLIC else None)
    y = sess.run(None, {"input": x.numpy()})
    print(f"onnxruntime: {y=}")


def main():
    inference()
    export()
    onnx_val()


if __name__ == "__main__":
    main()
