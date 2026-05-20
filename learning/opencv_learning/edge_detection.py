"""
Edge Detection 边缘检测
"""

import cv2

from Ahri.Asuka import SOKYOEI_DATA_DIR

WIN_NAME = "Canny"

min_threshold = 10
max_threshold = 100

img = cv2.imread(str(SOKYOEI_DATA_DIR / "Ahri/Popstar Ahri.jpg"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0.1)


def change_min_threshold(min_threshold):
    max_threshold = cv2.getTrackbarPos("max threshold", WIN_NAME)
    dst = cv2.Canny(img, min_threshold, max_threshold)
    cv2.imshow(WIN_NAME, dst)


def change_max_threshold(max_threshold):
    min_threshold = cv2.getTrackbarPos("min threshold", WIN_NAME)
    dst = cv2.Canny(img, min_threshold, max_threshold)
    cv2.imshow(WIN_NAME, dst)


def main():
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_FREERATIO)
    cv2.createTrackbar("min threshold", WIN_NAME, 10, 255, change_min_threshold)
    cv2.createTrackbar("max threshold", WIN_NAME, 20, 255, change_max_threshold)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
