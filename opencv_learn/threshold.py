"""
threshold 阈值
"""

import cv2
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show

PopstarAhriGray = cv2.cvtColor(PopstarAhri, cv2.COLOR_BGR2GRAY)


@img_show("threshold")
def threshold() -> MatLike:
    """全局阈值"""
    _, img = cv2.threshold(PopstarAhriGray, 127, 255, cv2.THRESH_BINARY)
    return img


@img_show("thresholdWithOtsu")
def thresholdWithOtsu() -> MatLike:
    """大津阈值"""
    _, img = cv2.threshold(PopstarAhriGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


@img_show("adaptiveThreshold")
def adaptiveThreshold() -> MatLike:
    """自适应阈值"""
    img = cv2.adaptiveThreshold(PopstarAhriGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


def main():
    threshold()
    thresholdWithOtsu()
    adaptiveThreshold()


if __name__ == '__main__':
    main()
