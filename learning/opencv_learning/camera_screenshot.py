"""
左键点击截取摄像头图像
"""

import datetime

import cv2

CAMERA_INDEX = 1  # 摄像头下标
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800


def main():
    def mouse_callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            cv2.imwrite(f"{time_text}.jpg", frame)
            print(f"save image to {time_text}.jpg")

    capture = cv2.VideoCapture(CAMERA_INDEX)

    # camera props
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    cv2.namedWindow("screenshot", cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("screenshot", mouse_callback)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        time_text = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        cv2.imshow("screenshot", frame)

        if cv2.waitKey(1) == 27:  # ESC quit
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
