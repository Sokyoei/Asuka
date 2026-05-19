import cv2
import tensorrt as trt
from cv2.typing import MatLike
from loguru import logger

from Ahri.Asuka import ASUKA_ROOT
from Ahri.Asuka.utils.tensorrt_utils import TensorRTModel
from Ahri.Asuka.utils.yolo_utils import COCO_CLASSES, plot_image, postprocess_ultralytics, preprocess


def yolov5su_example():
    model = TensorRTModel(ASUKA_ROOT / "models/yolov5su.engine")
    image: MatLike = cv2.imread(ASUKA_ROOT / "images/bus.jpg")
    height, width, _ = image.shape
    process_image = preprocess(image)
    result = model.inference(process_image)
    post_result = postprocess_ultralytics(result, width, height)
    plot_image(post_result, image, COCO_CLASSES)


def main():
    logger.info(f"TensorRT version: {trt.__version__}")
    yolov5su_example()


if __name__ == '__main__':
    main()
