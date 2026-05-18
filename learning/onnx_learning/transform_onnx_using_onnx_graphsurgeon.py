"""
使用 onnx_graphsurgeon 更改（融合）网络节点
"""

from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import torch


@gs.Graph.register()
def min(self: gs.Graph, *args):
    return self.layer(op="Min", inputs=args, outputs=["min_out_gs"])


@gs.Graph.register()
def max(self: gs.Graph, *args):
    return self.layer(op="Max", inputs=args, outputs=["max_out_gs"])


@gs.Graph.register()
def identity(self: gs.Graph, a):
    return self.layer(op="Identity", inputs=[a], outputs=["identity_out_gs"])


@gs.Graph.register()
def clip(self: gs.Graph, inputs, outputs):
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)


def create_onnx():
    graph = gs.Graph(opset=12)

    min_val = np.array(0, dtype=np.float32)
    max_val = np.array(1, dtype=np.float32)
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(5, 5))

    identity0 = graph.identity(input0)
    min0 = graph.min(*identity0, max_val)
    max0 = graph.max(*min0, min_val)
    output0 = graph.identity(*max0)

    graph.inputs = [input0]
    graph.outputs = output0

    for out in graph.outputs:
        out.dtype = np.float32

    onnx.save(gs.export_onnx(graph), "transform-minmax.onnx")


def transform_onnx():
    graph = gs.import_onnx(onnx.load("transform-minmax.onnx"))
    tensors: OrderedDict[str, gs.Tensor] = graph.tensors()

    inputs = [
        tensors["identity_out_gs_0"],
        tensors["onnx_graphsurgeon_constant_5"],
        tensors["onnx_graphsurgeon_constant_2"],
    ]
    outputs = [tensors["max_out_gs_6"]]

    # 将 min max 算子与网络断开
    for i in inputs:
        i.outputs.clear()
    for i in outputs:
        i.inputs.clear()

    graph.clip(inputs, outputs)
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "transform-clip.onnx")


def onnx_val(onnx_path, x: torch.Tensor):
    sess = ort.InferenceSession(onnx_path)
    y = sess.run(None, {"input0": x.numpy()})
    print(f"{x=}")
    print(f"{y=}")


def main():
    x = torch.rand(5, 5) * 10 - 5
    create_onnx()
    onnx_val("transform-minmax.onnx", x)
    transform_onnx()
    onnx_val("transform-clip.onnx", x)


if __name__ == "__main__":
    main()
