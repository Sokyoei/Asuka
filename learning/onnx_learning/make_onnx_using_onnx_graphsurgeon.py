"""
使用 onnx-graphsurgeon 创建 proto

Reference:
    https://github.com/kalfazed/tensorrt_starter/blob/main/chapter3-tensorrt-basics-and-onnx/3.5-onnxsurgeon/src/gs_create_conv.py
"""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

rng = np.random.default_rng()


def gs_conv():
    input = gs.Variable(name="input0", dtype=np.float32, shape=(1, 3, 224, 224))
    weight = gs.Constant(name="conv1.weight", values=rng.standard_normal(5, 3, 3, 3))
    bias = gs.Constant(name="conv1.bias", values=rng.standard_normal(5))
    output = gs.Variable(name="output0", dtype=np.float32, shape=(1, 5, 224, 224))
    node = gs.Node(op="Conv", inputs=[input, weight, bias], outputs=[output], attrs={"pads": [1, 1, 1, 1]})
    graph = gs.Graph(nodes=[node], inputs=[input], outputs=[output])
    model = gs.export_onnx(graph)
    onnx.save(model, "gs-conv.onnx")


@gs.Graph.register()
def add(self: gs.Graph, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])


@gs.Graph.register()
def mul(self: gs.Graph, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])


@gs.Graph.register()
def gemm(self: gs.Graph, a, b, trans_a=False, trans_b=False):
    attrs = {"tranA": int(trans_a), "tranB": int(trans_b)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)


@gs.Graph.register()
def relu(self: gs.Graph, a):
    return self.layer(op="Relu", inputs=[a], outputs=["relu_out_gs"])


def gs_register():
    graph = gs.Graph(opset=12)
    cons_a = gs.Constant(name="cons_a", values=rng.standard_normal(64, 32))
    cons_b = gs.Constant(name="cons_b", values=rng.standard_normal(64, 32))
    cons_c = gs.Constant(name="cons_c", values=rng.standard_normal(64, 32))
    cons_d = gs.Constant(name="cons_d", values=rng.standard_normal(64, 32))
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(64, 64))

    gemm0 = graph.gemm(input0, cons_a, trans_b=True)
    relu0 = graph.relu(*graph.add(*gemm0, cons_b))
    mul0 = graph.mul(*relu0, cons_c)
    output0 = graph.add(*mul0, cons_d)

    graph.inputs = [input0]
    graph.outputs = output0

    for out in graph.outputs:
        out.dtype = np.float32

    onnx.save(gs.export_onnx(graph), "gs-register.onnx")


def main() -> None:
    gs_conv()
    gs_register()


if __name__ == "__main__":
    main()
