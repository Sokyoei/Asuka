"""
YOLOv5 ONNXRuntime

```shell
git clone https://github.com/ultralytics/yolov5
cd yolov5
conda create -n yolo python=3.9
pip install -r requirements.txt
python export.py --weights model_path --include=onnx
```
"""

from typing import List

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray

from Ahri.Asuka import ASUKA_ROOT
from Ahri.Asuka.utils import nms, plot_image, xywh_to_xyxy

# fmt:off
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
# fmt:on


class YOLOv5ONNX(object):

    def __init__(self, onnx_path, conf_threshold: float, nms_threshold) -> None:
        model = onnx.load(str(onnx_path))
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        try:
            onnx.checker.check_model(model)
        except Exception:
            print("model error")

        self.options = ort.SessionOptions()
        self.options.enable_profiling = True
        providers: List[str] = ort.get_available_providers()
        # 删除 TensorrtExecutionProvider
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        self.ort_session = ort.InferenceSession(onnx_path, self.options, providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def inference(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, [640, 640])
        # BGR -> RGB, HWC -> CHW
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img = img.astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        preds: List[NDArray] = self.ort_session.run([self.output_name], {self.input_name: img})
        return preds, img

    def postprocess(self, preds: List[NDArray]):
        """后处理，去除多余的候选框"""
        for pred in preds:
            pred = pred.squeeze()
            box = pred[pred[:, 4] > self.conf_threshold]  # 去除置信度低于 conf_threshold 的框
            classes = np.argmax(box[:, 5:], axis=-1)
            classes = np.expand_dims(classes, axis=-1)
            box_xyxy = xywh_to_xyxy(box[:, :5])
            box_classes = np.concatenate((box_xyxy, classes), axis=-1)
            keep = nms(box_classes, self.nms_threshold)
            return box_classes[keep]


def main():
    yolov5 = YOLOv5ONNX(ASUKA_ROOT / "models/yolov5s.onnx", 0.5, 0.7)
    preds, img = yolov5.inference(ASUKA_ROOT / "images/bus.jpg")
    result = yolov5.postprocess(preds)
    img = img.squeeze().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plot_image(result, img, CLASSES)


if __name__ == "__main__":
    main()
