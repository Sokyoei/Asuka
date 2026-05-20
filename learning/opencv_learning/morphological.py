"""
形态学操作
"""

import cv2
import numpy as np
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show


@img_show("erode")
def erode(img: MatLike) -> MatLike:
    """腐蚀"""
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dst = cv2.erode(img, kernel)
    return dst


@img_show("dilate")
def dilate(img: MatLike) -> MatLike:
    """膨胀"""
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dst = cv2.dilate(img, kernel)
    return dst


@img_show("open")
def open_(img: MatLike) -> MatLike:
    """开运算：先腐蚀后膨胀，用来移除由图像噪音形成的斑点"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return dst


@img_show("close")
def close(img: MatLike) -> MatLike:
    """闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return dst


@img_show("gradient")
def gradient(img: MatLike) -> MatLike:
    """形态学梯度"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return dst


@img_show("tophat")
def tophat(img: MatLike) -> MatLike:
    """
    顶帽，又称礼帽

    tophat = open - src
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return dst


@img_show("blackhat")
def blackhat(img: MatLike) -> MatLike:
    """
    黑帽

    blackhat = close - src
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return dst


def main():
    erode(PopstarAhri)
    dilate(PopstarAhri)
    open_(PopstarAhri)
    close(PopstarAhri)
    gradient(PopstarAhri)
    tophat(PopstarAhri)
    blackhat(PopstarAhri)


if __name__ == '__main__':
    main()
