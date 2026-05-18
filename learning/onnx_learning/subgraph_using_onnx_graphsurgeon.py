"""
export torchvision.models.swin_t to onnx
"""

from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def main():
    model = onnx.load("SwinTransformer.onnx")

    # LayerMorm
    graph = gs.import_onnx(model)
    tensors = graph.tensors()
    print(tensors["/features/features.0/features.0.1/Transpose_output_0"])
    print(tensors["/features/features.0/features.0.2/Div_output_0"])
    graph.inputs = [
        tensors["/features/features.0/features.0.1/Transpose_output_0"].to_variable(
            dtype=np.float32, shape=(1, 56, 56, 96)
        )
    ]
    graph.outputs = [
        tensors["/features/features.0/features.0.2/Div_output_0"].to_variable(
            dtype=np.float32, shape=(1, 1, 56, 56, 96)
        )
    ]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "swin_t-layernorm.onnx")

    # MHSA
    graph = gs.import_onnx(model)
    tensors: OrderedDict[str, gs.Tensor] = graph.tensors()
    print(tensors["/features/features.1/features.1.0/attn/Reshape_3_output_0"])
    print(tensors["/features/features.1/features.1.0/attn/MatMul_3_output_0"])
    graph.inputs = [
        tensors["/features/features.1/features.1.0/attn/Reshape_3_output_0"].to_variable(
            dtype=np.float32, shape=(64, 49, 96)
        )
    ]
    graph.outputs = [
        tensors["/features/features.1/features.1.0/attn/MatMul_3_output_0"].to_variable(
            dtype=np.float32, shape=(64, 49, 96)
        )
    ]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "swin_t-mhsa.onnx")


if __name__ == "__main__":
    main()
