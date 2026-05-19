# [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)

## 流程

- Logger    日志记录器
- Builder   用于创建 Network, 对模型序列化生成 engine
- Network   由 Builder 创建，最初是一个空容器
- Parser    用于解析 onnx 模型
- Context   上接 engine, 下接 inference

## 参考

- [docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)
- [trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)
- [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

## .onnx 转换 .engine

### trtexec

```shell
trtexec --onnx=onnx_file\
        --memPoolSize=workspace:2048\
        --saveEngine=engine_file\
        --verbose\
        --profilingVerbosity=detailed\
        --dumpOutput\
        --dumpProfile\
        --dumpLayerInfo\
        --exportOutput=build_output.log\
        --exportProfile=build_profile.log\
        --exportLayerInfo=build_layer_info.log\
        --iterations=50\
        --fp16
```

## FLOPS/TOPS

FLOPS(Floating point number operators per second) 一秒钟可以处理浮点运算的次数
TOPS(Tera operators per second) 一秒钟可以处理整形运算的次数

FLOPS = 时钟频率 * Core 数量 * 每个时钟周期可以处理的FLOPS

## Roolfline model

## Quantization「量化」

量化针对的是 activation value 和 weight

PTQ(Post-Training Quantization)
QAT(Quantization-Aware Training)

Calibration

- Minmax Calibration
- Entropy Calibration
- Percentile Calibration

- pre-tensor
- pre-layer

## Prunning「剪枝」
