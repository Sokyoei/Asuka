"""
多人人体姿态检测
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy.typing import NDArray

from Ahri.Asuka import ASUKA_ROOT

MARGIN = 20  # pixels
ROW_SIZE = 20  # pixels
FONT_SIZE = 2
TEXT_COLOR = (255, 0, 0)  # red

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, cv2.LINE_4)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, cv2.LINE_4)

    return image


def inference(image_path):
    # download https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    pose_detector = mp_pose.Pose(
        static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5
    )

    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    image_copy: NDArray = np.copy(image.numpy_view())
    src_h, src_w, _ = image_copy.shape
    annotated_image = visualize(image_copy, detection_result)
    for detection in detection_result.detections:
        x = detection.bounding_box.origin_x
        y = detection.bounding_box.origin_y
        w = detection.bounding_box.width
        h = detection.bounding_box.height
        # check index out of bounds
        image_pose = image_copy[
            y if y > 0 else 0 : src_h if y + h > src_h else y + h,
            x if x > 0 else 0 : src_w if x + w > src_w else x + w,
            :,
        ]
        # image_pose = image_copy[y:y+h, x:x+w, :]

        results = pose_detector.process(image_pose)
        for i in results.pose_landmarks.landmark:
            src_x = int(i.x * w + x)
            src_y = int(i.y * h + y)
            cv2.circle(image_copy, (src_x, src_y), 2, (0, 255, 0), 2)

    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Object Detection", cv2.WINDOW_FREERATIO)
    cv2.imshow("Object Detection", bgr_annotated_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    inference(str(ASUKA_ROOT / r"data\Ahri\Popstar Ahri.jpg"))


if __name__ == "__main__":
    main()
