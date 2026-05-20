"""
霍夫直线、圆检测
"""

import cv2
import numpy as np
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import GREEN, PopstarAhri, img_show


@img_show("HoughLines")
def HoughLines(img: MatLike):
    plot_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 150, 200, 3)
    lines = cv2.HoughLines(canny, 1, np.pi / 180.0, 200)
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        cv2.line(plot_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return plot_img


@img_show("HoughLinesP")
def HoughLinesP(img: MatLike):
    plot_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 150, 200, 3)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180.0, 150, 200, 5)
    for [[x1, y1, x2, y2]] in lines:
        cv2.line(plot_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return plot_img


@img_show("HoughCircles")
def HoughCircles(img: MatLike):
    plot_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=40)
    circles = circles.astype(np.int32)
    if circles.size:
        for x, y, radius in circles[0]:
            cv2.circle(plot_img, [x, y], radius, GREEN, 2)
    return plot_img


def main():
    HoughLines(PopstarAhri)
    HoughLinesP(PopstarAhri)
    HoughCircles(PopstarAhri)


if __name__ == '__main__':
    main()
