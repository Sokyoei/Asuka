"""
Dataset bbox:

    pascal voc:       [x_min, y_min, x_max, y_max]
    coco:             [x_min, y_min, width, height]
    yolo(normalized): [x_center, y_center, width, height]
"""

import cv2
import numpy as np
from numpy.typing import NDArray

# fmt:off
COCO_CLASSES = [
    "person",           "bicycle",      "car",              "motorcycle",       "airplane",
    "bus",              "train",        "truck",            "boat",             "traffic light",
    "fire hydrant",     "stop sign",    "parking meter",    "bench",            "bird",
    "cat",              "dog",          "horse",            "sheep",            "cow",
    "elephant",         "bear",         "zebra",            "giraffe",          "backpack",
    "umbrella",         "handbag",      "tie",              "suitcase",         "frisbee",
    "skis",             "snowboard",    "sports ball",      "kite",             "baseball bat",
    "baseball glove",   "skateboard",   "surfboard",        "tennis racket",    "bottle",
    "wine glass",       "cup",          "fork",             "knife",            "spoon",
    "bowl",             "banana",       "apple",            "sandwich",         "orange",
    "broccoli",         "carrot",       "hot dog",          "pizza",            "donut",
    "cake",             "chair",        "couch",            "potted plant",     "bed",
    "dining table",     "toilet",       "tv",               "laptop",           "mouse",
    "remote",           "keyboard",     "cell phone",       "microwave",        "oven",
    "toaster",          "sink",         "refrigerator",     "book",             "clock",
    "vase",             "scissors",     "teddy bear",       "hair drier",       "toothbrush",
]
# fmt:on
INPUT_WIDTH = 640
INPUT_HEIGHT = 640


def preprocess(image: NDArray) -> NDArray:
    return cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)


def postprocess():
    pass


def postprocess_ultralytics(
    yolo_results: NDArray,
    original_width: int,
    original_height: int,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.45,
):
    dets = np.array([])
    dets = np.squeeze(yolo_results)
    dets = dets[np.any(dets != 0, axis=1)]

    dets[:, 0] = dets[:, 0] / INPUT_WIDTH * original_width
    dets[:, 1] = dets[:, 1] / INPUT_HEIGHT * original_height
    dets[:, 2] = (dets[:, 0] + dets[:, 2]) / INPUT_WIDTH * original_width
    dets[:, 3] = (dets[:, 1] + dets[:, 3]) / INPUT_HEIGHT * original_height

    return dets


def xywh_to_xyxy(x) -> NDArray:
    """(x,y,w,h) -> (x1,y1,x2,y2)

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def plot_image(yolo_results: NDArray, image: NDArray, CLASSES: list[str]):
    cv2.namedWindow("img", cv2.WINDOW_FREERATIO)

    for x1, y1, x2, y2, score, classes_index in yolo_results:
        x1, y1, x2, y2, classes_index = int(x1), int(y1), int(x2), int(y2), int(classes_index)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{CLASSES[classes_index]} {score:.2}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nms(dets: NDArray, thresh: float):
    """
    NMS(Non-Max Supperssion) 非极大值抑制

    Args:
        dets (ndarray): [[x1,y1,x2,y2,score,class],...]
        thresh (float): thresh
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep
