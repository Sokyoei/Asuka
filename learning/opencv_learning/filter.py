"""
filter 滤波
"""

import cv2
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show


@img_show("mean_blur")
def mean_blur(img: MatLike) -> MatLike:
    dst = cv2.blur(img, (3, 3))
    return dst


@img_show("median_blur")
def median_blur(img: MatLike) -> MatLike:
    dst = cv2.medianBlur(img, 3)
    return dst


@img_show("stack_blur")
def stack_blur(img: MatLike) -> MatLike:
    dst = cv2.stackBlur(img, (3, 3))
    return dst


@img_show("gaussian_blur")
def gaussian_blur(img: MatLike) -> MatLike:
    dst = cv2.GaussianBlur(img, (3, 3), 0.1)
    return dst


def main():
    mean_blur(PopstarAhri)
    median_blur(PopstarAhri)
    stack_blur(PopstarAhri)
    gaussian_blur(PopstarAhri)


if __name__ == "__main__":
    main()
