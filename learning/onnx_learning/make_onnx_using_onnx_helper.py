"""
使用 onnx.helper 创建 proto

Reference:
    https://github.com/kalfazed/tensorrt_starter/blob/main/chapter3-tensorrt-basics-and-onnx/3.3-read-and-parse-onnx/src/create_onnx_linear.py
"""

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto, TensorProto, helper

rng = np.random.default_rng()


def create_onnx() -> ModelProto:
    # 创建 ValueProto
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])

    # 创建 NodeProto
    mul = helper.make_node('Mul', ['a', 'x'], 'c', "multiply")
    add = helper.make_node('Add', ['c', 'b'], 'y', "add")

    # 构建 GraphProto
    graph = helper.make_graph([mul, add], 'sample-linear', [a, x, b], [y])

    # 构建 ModelProto
    model = helper.make_model(graph)

    # 检查 model 是否有错误
    onnx.checker.check_model(model)

    # 保存 model
    onnx.save(model, "sample-linear.onnx")

    return model


def create_init_tensor(name: str, tensor_array: NDArray, data_type: TensorProto = TensorProto.FLOAT):
    return helper.make_tensor(name, data_type, tensor_array.shape, tensor_array.flatten().tolist())


def create_net() -> ModelProto:
    input_name = "input"
    output_name = "output"

    input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 64, 64])
    output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 16, 1, 1])

    conv2d_1_name = "conv2d_1"
    conv2d_1_weight_name = "conv2d_1.weight"
    conv2d_1_bias_name = "conv2d_1.bias"
    conv2d_1_output_name = "conv2d_1.output"

    conv2d_1_weight_init = create_init_tensor(conv2d_1_weight_name, rng.standard_normal(32, 64, 3, 3))
    conv2d_1_bias_init = create_init_tensor(conv2d_1_bias_name, rng.standard_normal(32))
    conv2d_1 = helper.make_node(
        "Conv",
        [input_name, conv2d_1_weight_name, conv2d_1_bias_name],
        [conv2d_1_output_name],
        conv2d_1_name,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    bn1_name = "batchnorm_1"
    bn1_scale_name = "batchnorm_1.scale"
    bn1_bias_name = "batchnorm_1,bias"
    bn1_mean_name = "batchnorm_1.mean"
    bn1_var_name = "batchnorm_1.var"
    bn1_output_name = "batchnorm_1.output"

    bn1_scale_init = create_init_tensor(bn1_scale_name, rng.standard_normal(32))
    bn1_bias_init = create_init_tensor(bn1_bias_name, rng.standard_normal(32))
    bn1_mean_init = create_init_tensor(bn1_mean_name, rng.standard_normal(32))
    bn1_var_init = create_init_tensor(bn1_var_name, rng.standard_normal(32))

    bn1 = helper.make_node(
        "BatchNormalization",
        [conv2d_1_output_name, bn1_scale_name, bn1_bias_name, bn1_mean_name, bn1_var_name],
        [bn1_output_name],
        bn1_name,
    )

    relu_name = "relu"
    relu_output_name = "relu.output"

    relu = onnx.helper.make_node("Relu", [bn1_output_name], [relu_output_name], relu_name)

    avg_pool_name = "avg_pool"
    avg_pool_output_name = "avg_pool.output"

    avg_pool = onnx.helper.make_node("GlobalAveragePool", [relu_output_name], [avg_pool_output_name], avg_pool_name)

    conv2d_2_name = "conv2d_2"
    conv2d_2_weight_name = "conv2d_2.weight"
    conv2d_2_bias_name = "conv2d_2.bias"

    conv2d_2_weight_init = create_init_tensor(conv2d_2_weight_name, rng.standard_normal(16, 32, 1, 1))
    conv2d_2_bias_init = create_init_tensor(conv2d_2_bias_name, rng.standard_normal(16))
    conv2d_2 = helper.make_node(
        "Conv",
        [avg_pool_output_name, conv2d_2_weight_name, conv2d_2_bias_name],
        [output_name],
        conv2d_2_name,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        [conv2d_1, bn1, relu, avg_pool, conv2d_2],
        "conv",
        initializer=[
            conv2d_1_weight_init,
            conv2d_1_bias_init,
            bn1_scale_init,
            bn1_bias_init,
            bn1_mean_init,
            bn1_var_init,
            conv2d_2_weight_init,
            conv2d_2_bias_init,
        ],
        inputs=[input],
        outputs=[output],
    )

    model = onnx.helper.make_model(graph, producer_name="onnx-sample")
    model.opset_import[0].version = 12

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, "sample-conv.onnx")

    return model


def main():
    model = create_onnx()
    print(model)
    model2 = create_net()
    print(model2)


if __name__ == "__main__":
    main()
