import cv2
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.engine.results import Results

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show

model = YOLO("yolov8n-pose.pt")
model.info()


def inference_video(index):
    capture = cv2.VideoCapture(index)

    cv2.namedWindow("ultralytics", cv2.WINDOW_FREERATIO)

    # main loop
    while True:
        ret, img = capture.read()
        if not ret:
            break

        # do inference
        result: Results = model(img)

        cv2.imshow("ultralytics", result[0].plot())
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


@img_show("ultralytics")
def inference_image(img: MatLike) -> MatLike:
    # do inference
    result: Results = model(img)
    return result[0].plot()


def main():
    inference_image(PopstarAhri)
    # inference_video(0)


if __name__ == "__main__":
    main()
