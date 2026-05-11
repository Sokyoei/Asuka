# [OpenMMLab](https://openmmlab.com)

## 环境搭建

```bash
# torch 2.1.x with mmcv 2.1.0 on Ubuntu 22.04
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# fix environment
pip install "numpy<2"  # mmcv 2.1.0 requires numpy<2
pip install opencv-python==4.11.0.86  #  next version requires numpy>2

# install with mim(or git clone using `pip install -e .` for editable install)
pip install -U openmim
mim install mmcv==2.1.0 --trusted-host download.openmmlab.com
mim install mmdet==3.2.0 --trusted-host download.openmmlab.com
mim install mmpose==1.3.2 --trusted-host download.openmmlab.com
```

## 模型转换

### 编译 MMDeploy

!!! warning TensorRT 8.6.1.6 兼容性好，~~好久没怎么大更新了~~，高版本会编译错误，需要修改部分代码

```bash
# build on Ubuntu 22.04
git clone https://github.com/open-mmlab/mmdeploy.git --recursive
cd mmdeploy

# build with backend trt and vcpkg
cmake -B build -DMMDEPLOY_TARGET_BACKENDS=trt -DMMDEPLOY_BUILD_SDK=ON -DTENSORRT_DIR=~/TensorRT-8.6.1.6 -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-linux
cd build
make -j$(nproc)
```

### 模型转换(RTMO-s)

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"~/TensorRT-8.6.1.6/lib"

# clip model
cd mmpose
python tools/misc/publish_model.py ../rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth ../rtmo-s_publish.pth

# convert model
cd mmdeploy
python tools/deploy.py configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py ../mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py ../rtmo-s_publish-41b20847_20250821.pth ../../../data/Ahri/Ahri.jpg --work-dir ../output --device cuda:0
```
