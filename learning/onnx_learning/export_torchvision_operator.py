import onnxruntime as ort
import torch
from numpy.typing import NDArray
from onnxruntime_extensions import PyCustomOpDef, onnx_op
from onnxruntime_extensions import get_library_path as _lib_path
from torch import Tensor, nn
from torch.onnx import register_custom_op_symbolic
from torch.onnx._internal import jit_utils
from torch.onnx.symbolic_helper import parse_args
from torchvision.ops import DeformConv2d, deform_conv2d

MODEL_CUSTOM_OP_NAME = "torchvision_deform_conv2d.onnx"


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
def deform_conv2d_symbolic(
    g: jit_utils.GraphContext,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask,
):
    # input_shape = input.type().sizes()
    # offset_shape = offset.type().sizes()
    y = g.op("Ahri::deform_conv2d", input, offset)
    # y.setType(x.type().with_sizes(input_shape))
    return y


register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, 12)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 18, 3)
        self.deform_conv = DeformConv2d(3, 3, 3)

    def forward(self, x):
        return self.deform_conv(x, self.conv(x))


model = Model()
x: Tensor = torch.randn(1, 3, 5, 5)


@onnx_op(
    op_type="Ahri::deform_conv2d",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def torchvision_deform_conv2d_ort(x: NDArray, offset: NDArray) -> NDArray:
    x = torch.tensor(x)
    print("---------------------------------------------------------------------------")
    off = torch.tensor(offset)
    print(f"{off.shape=}")
    offset = torch.conv2d(off)
    print(f"{    offset.shape=}")

    print("---------------------------------------------------------------------------")

    y: Tensor = deform_conv2d(x, offset)
    print("---------------------------------------------------------------------------")

    return y.numpy()


so = ort.SessionOptions()
so.register_custom_ops_library(_lib_path())


def inference():
    model.eval()
    y: Tensor = model(x)
    print(f"{x.data=}")
    print(f"torch: {y.data=}")
    print()


def export():
    torch.onnx.export(
        model, (x,), MODEL_CUSTOM_OP_NAME, input_names=["input"], output_names=["output"], opset_version=12
    )


def onnx_val():
    model.eval()
    sess = ort.InferenceSession(MODEL_CUSTOM_OP_NAME, so)
    y = sess.run(None, {"input": x.numpy()})
    print(f"onnxruntime: {y=}")


def main():
    inference()
    export()
    onnx_val()


if __name__ == "__main__":
    main()
