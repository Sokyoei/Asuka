# ONNX

ONNX(Open Neural Network Exchange)

[docs](https://onnx.ai/onnx/index.html)

> ONNX 模型是使用 Protobuf 序列化格式的文件

## onnx 模型格式

onnx 中的各类型 proto 定义 [onnx.in.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto)

```text
ModelProto                              整个模型的信息
└───GraphProto                          整个网络的信息
    ├───NodeProto(node)                 各个计算节点的信息，如 Conv, Linear
    ├───ValueInfoProto(input/output)    模型输入输出节点的信息
    └───TensorProto(initializer)        模型权重的信息
```

onnx 中的各类型算子支持的版本(或修改) [Operators.md](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

## ONNXRuntime

[docs](https://onnxruntime.ai/docs/)

> ONNX 模型部署

## 模型转换

```bash
pip install torch
pip install tf2onnx
pip install skl2onnx
```
