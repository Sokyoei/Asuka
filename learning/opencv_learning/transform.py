"""
几何变换
    仿射变换
    透视变换
"""

import cv2
import numpy as np
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, show_image


def affine(img: MatLike) -> MatLike:
    """仿射变换"""
    height, width, _ = img.shape
    M = cv2.getAffineTransform(
        np.float32([[0, 0], [0, height], [width, 0]]), np.float32([[0, 0], [width / 2, height], [width, 0]])
    )
    dst = cv2.warpAffine(img, M, (width, height))
    show_image(dst, "warpAffine")
    return dst, M


def affine_inv(img: MatLike, M) -> MatLike:
    """反向仿射变换"""
    height, width, _ = img.shape
    # 计算逆变换矩阵
    M_inv = cv2.invertAffineTransform(M)
    # 应用逆仿射变换
    dst = cv2.warpAffine(img, M_inv, (width, height))
    show_image(dst, "warpAffine_inv")
    return dst


def perspective(img: MatLike) -> MatLike:
    """透视变换"""
    height, width, _ = img.shape
    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [0, height], [width, 0], [width, height]]),
        np.float32([[width / 2, 0], [0, height], [width, 0], [width / 2, height]]),
    )
    dst = cv2.warpPerspective(img, M, (width, height))
    show_image(dst, "warpPerspective")
    return dst, M


def perspective_inv(img: MatLike, M) -> MatLike:
    """反向透视变换"""
    height, width, _ = img.shape
    # 计算逆变换矩阵
    M_inv = np.linalg.inv(M)
    # 应用逆透视变换
    dst = cv2.warpPerspective(img, M_inv, (width, height))
    show_image(dst, "warpPerspective_inv")
    return dst


def main():
    img_affine = cv2.resize(PopstarAhri, (0, 0), None, 0.2, 0.2)
    affine_result, M = affine(img_affine)
    affine_inv(affine_result, M)

    img_perspective = cv2.resize(PopstarAhri, (0, 0), None, 0.2, 0.2)
    perspective_result, M = perspective(img_perspective)
    perspective_inv(perspective_result, M)


if __name__ == "__main__":
    main()
