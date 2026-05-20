"""
直方图
"""

import cv2
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show


@img_show("Ahri_equalizeHist")
def Ahri_equalizeHist(img: MatLike):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eH = cv2.equalizeHist(img_gray)
    return img_eH


@img_show("Ahri_CLAHE")
def Ahri_CLAHE(img: MatLike):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE()
    img_clahe = clahe.apply(img_gray)
    return img_clahe


def main():
    Ahri_equalizeHist(PopstarAhri)
    Ahri_CLAHE(PopstarAhri)


if __name__ == '__main__':
    main()
