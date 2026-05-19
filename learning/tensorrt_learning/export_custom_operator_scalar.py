import onnx
import onnxruntime as ort
import onnxsim
import torch
from numpy.typing import NDArray
from onnxruntime_extensions import PyCustomOpDef, onnx_op
from onnxruntime_extensions import get_library_path as _lib_path  # noqa: F401
from torch import Tensor, nn
from torch.autograd.function import Function, FunctionCtx
from torch.onnx._internal import jit_utils

MODEL_CUSTOM_OP_NAME = "samples_scalar_net.onnx"


class SimplesScalarImpl(Function):

    @staticmethod
    def symbolic(g: jit_utils.GraphContext, x: torch.Value, r, s) -> torch.Value:
        """注册算子，可以将散列的一些算子整合成一个算子"""
        x_shape = x.type().sizes()
        y_shape = x_shape
        y = g.op("samples::Scalar", x, scalar_f=r, scale_f=s)
        y.setType(x.type().with_sizes(y_shape))
        return y

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, r, s) -> Tensor:
        ctx.save_for_backward(x)
        return (x + r) * s


class SimplesScalar(nn.Module):

    def __init__(self, r, s):
        super().__init__()
        self.r = r
        self.s = s

    def forward(self, x):
        return SimplesScalarImpl.apply(x, self.r, self.s)


class ScalarNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, (3, 3), padding=1)
        self.act = SimplesScalar(1, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.05)
                nn.init.constant_(m.bias, 0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


model = ScalarNet()

x = torch.tensor(
    [
        [
            [
                [0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
                [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
                [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
                [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
                [0.3138, 0.1980, 0.4162, 0.2843, 0.3398],
            ]
        ]
    ]
)


@onnx_op(
    op_type="simples::Scalar",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def simples_scalar_ort(x: NDArray, r: NDArray, s: NDArray) -> NDArray:
    return (x + r) * s


# so = ort.SessionOptions()
# so.register_custom_ops_library(_lib_path())


def inference():
    model.eval()
    y: Tensor = model(x)
    print(f"{x.data=}")
    print(f"torch: {y.data=}")


def export():
    torch.onnx.export(
        model,
        (x,),
        MODEL_CUSTOM_OP_NAME,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        custom_opsets={"samples": 1},
    )
    model_onnx = onnx.load(MODEL_CUSTOM_OP_NAME)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "simplify failed"
    onnx.save(model_onnx, MODEL_CUSTOM_OP_NAME)


def onnx_val():
    model.eval()
    sess = ort.InferenceSession(MODEL_CUSTOM_OP_NAME)
    y = sess.run(None, {"input": x.numpy()})
    print(f"onnxruntime: {y=}")


def main():
    inference()
    export()
    # onnx_val()


if __name__ == "__main__":
    main()
