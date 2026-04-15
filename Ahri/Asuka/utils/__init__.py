from Ahri.Asuka.check import check_package_installed

__all__ = []

if check_package_installed("onnxruntime"):
    from .onnx_utils import ONNXRuntimeModel

    __all__ += ["ONNXRuntimeModel"]

# NOTE: tensorflow_utils.py 单独使用，因为初始化后会导入 tensorflow, 速度有些慢
# if check_package_installed("tensorflow") and check_package_installed("keras"):
#     from .tensorflow_utils import DEVICE as TF_DEVICE

#     KERAS_DEVICE = TF_DEVICE

#     __all__ += ["KERAS_DEVICE", "TF_DEVICE"]

if check_package_installed("tensorrt"):
    from .tensorrt_utils import TensorRTModel

    __all__ += ["TensorRTModel"]

if check_package_installed("openvino"):
    from .openvino_utils import OpenVINOModel

    __all__ += ["OpenVINOModel"]

if check_package_installed("torch"):
    from .torch_utils import DEVICE, AbstractTorchDataset

    TORCH_DEVICE = DEVICE

    __all__ += ["DEVICE", "TORCH_DEVICE", "AbstractTorchDataset"]


from .yolo_utils import nms, plot_image, xywh_to_xyxy

__all__ += ["nms", "plot_image", "xywh_to_xyxy"]
