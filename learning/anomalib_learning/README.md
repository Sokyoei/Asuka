# [anomalib](https://github.com/open-edge-platform/anomalib)

Anomaly Detection Library 工业异常检测库

## 安装

```shell
pip install anomalib[full]
```

## 训练

数据集格式

```text
dataset
├───train
│   └───good
│       ├───00000001.jpg
│       ├───00000002.jpg
│       └───...
└───test
    ├───good
    │   ├───00000001.jpg
    │   ├───00000002.jpg
    │   └───...
    └───defect
        ├───00000001.jpg
        ├───00000002.jpg
        └───...
```
