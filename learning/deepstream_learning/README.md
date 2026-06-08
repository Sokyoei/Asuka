# DeepStream

[DeepStream](https://developer.nvidia.com/deepstream-sdk)

## docker 开发环境

宿主机

```shell
docker pull nvcr.io/nvidia/deepstream:8.0-triton-multiarch
# see https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream

# X11 授权问题
xhost +
```

容器

```shell
# WARNING: 注意宿主机和容器的 $DISPLAY 值需要一样才能正常显示图形
# 查看 DeepStream 版本
deepstream-app --version-all
# 查看 nvidia-smi 是否可用
nvidia-smi
# [可选]安装音频库
bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh
# 运行测试案例(文件名可能根据版本不同而不同)
cd /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app
deepstream-app -c source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display.txt

## pyds 安装
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl
pip install ./pyds-1.2.2-cp312-cp312-linux_x86_64.whl
```

## YOLO 集成

```shell
# 构建 libnvdsinfer_custom_impl_Yolo.so
cd /workspace/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
export CUDA_VER=12.8
make

# 转换模型
cd /workspace/DeepStream-Yolo/utils
python3 export_yolo26.py -w yolo26l.pt --simplify --batch 40 --opset 18

# WARNING: 按照 `DeepStream-Yolo` 运行案例时工作路径需要在 `DeepStream-Yolo` 路径下（或者自己组织文件夹结构）
cd /workspace/DeepStream-Yolo
deepstream-app -c deepstream_app_config.txt
```
