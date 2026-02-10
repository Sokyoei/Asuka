# 三维重建

## 三维几何表示

- 顶点表示
- 多边形网格表示
- 参数曲面表示
- 体素表示

## SfM

Structure from Motion「运动恢复结构」

## MVS

Multi-View Stereo「多视图立体视觉」

[openMVS](https://github.com/cdcseacave/openMVS)

## NeRF

Neural Radiance Fields「神经辐射场」

- [awesome-NeRF](https://github.com/awesome-NeRF/awesome-NeRF)
- [nerf](https://github.com/bmild/nerf)
- [paper](https://arxiv.org/abs/2003.08934)
- [website](https://www.matthewtancik.com/nerf)
- [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

### ?

- 输入 5D 向量（）
- 输出 4D 向量（密度 颜色（RGB））

## 3DGS

3D Gaussian Splatting「3D高斯飞溅」

- [awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [website](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

特点：机器学习、CG、线性代数

### 渲染

- Ray-casting(NeRF), 被动，计算出每个像素点受到的发光粒子的影响生成图像
- Splatting(3DGS), 主动，计算每个发光粒子如何影响像素点

椭球高斯公式：
$$
G(x)=
\frac{1}{\sqrt{(2\pi)^k\begin{vmatrix}\Sigma\end{vmatrix}}}
e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$
$\Sigma$ 是协方差矩阵，半正定，$\begin{vmatrix}\Sigma\end{vmatrix}$是其行列式

### 使用

数据集结构

```txt
data
└───input               # 一系列图片文件
    ├───database.db     # colmap 数据库文件
    └───spare
        └───0
            ├───cameras.bin
            ├───images.bin
            ├───points3D.bin
            └───project.ini
```

下载 [colmap](https://github.com/colmap/colmap)，用于特征提取和重建稀疏点云

### tools

- [SIBR_viewer](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)
- [SuperSplat](https://playcanvas.com/supersplat/editor/)

运行 SIBR_viewer

```shell
SIBR_gaussianViewer_app -m your_model_path
```

球谐函数
