# TensorRT FAQ

## Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors

```log
[10/09/2024-11:33:10] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
```

TensorRT 模型在另一台机器上构建的，最好在每台机器上单独构建对应的 TensorRT 模型。如果确认生成和运行环境严格一致，可以忽略此警告。
