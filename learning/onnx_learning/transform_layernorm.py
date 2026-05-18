import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime
import onnxsim
import torch
from torch import nn

TORCH_EXPORT_ONNX = "layernorm-pytroch.onnx"
MERGE_ONNX = "layernorm-merge.onnx"


@gs.Graph.register()
def identity(self: gs.Graph, inputs, outputs):
    return self.layer(op="Identity", inputs=inputs, outputs=outputs)


@gs.Graph.register()
def layernorm(self: gs.Graph, inputs, outputs, axis, epsilon):
    attrs = {'axis': np.int64(axis), 'epsilon': float(epsilon)}
    return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs, attrs=attrs)


@gs.Graph.register()
def layernorm_default(self: gs.Graph, inputs, outputs):
    return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(3)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        _, _, H, W = x.shape
        L = H * W
        x = self.conv1(x)
        x = x.view(x.shape[0], x.shape[1], L).permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        return x


def export_onnx():
    input = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)
    model = Model()
    model.eval()

    torch.onnx.export(
        model=model,
        args=(input,),
        f=TORCH_EXPORT_ONNX,
        input_names=["input0"],
        output_names=["output0"],
        opset_version=12,
    )

    model_onnx = onnx.load(TORCH_EXPORT_ONNX)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, TORCH_EXPORT_ONNX)


def transform_onnx():
    graph = gs.import_onnx(onnx.load_model(TORCH_EXPORT_ONNX))
    tensors = graph.tensors()

    norm_scale = gs.Constant(name="norm.weight", values=np.ones(shape=[3], dtype=np.float32))
    norm_bias = gs.Constant(name="norm.bias", values=np.zeros(shape=[3], dtype=np.float32))

    inputs = [tensors["/Transpose_output_0"]]
    outputs = [tensors["/norm/Div_output_0"]]

    for item in inputs:
        item.outputs.clear()

    for item in outputs:
        item.inputs.clear()

    inputs = [tensors["/Transpose_output_0"], norm_scale, norm_bias]
    epsilon = [tensors["/norm/Constant_1_output_0"]]
    print(type(epsilon[0].values))

    graph.layerNorm(inputs, outputs, axis=-1, epsilon=epsilon[0].values)
    # graph.identity(inputs, outputs)
    # graph.layerNorm_default(inputs, outputs)

    graph.cleanup()

    onnx.save(gs.export_onnx(graph), MERGE_ONNX)


def validate_onnx(onnx_path: str | os.PathLike, input: torch.Tensor):
    sess = onnxruntime.InferenceSession(onnx_path)
    output = sess.run(None, {'input0': input.numpy()})

    print(f"{input=}")
    print(f"{output=}")


def main() -> None:
    input = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)

    export_onnx()
    transform_onnx()

    validate_onnx(TORCH_EXPORT_ONNX, input)
    validate_onnx(MERGE_ONNX, input)


if __name__ == "__main__":
    main()
