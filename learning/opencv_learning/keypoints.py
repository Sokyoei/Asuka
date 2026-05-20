"""
keypoints 关键点检测
"""

from typing import Literal

import cv2
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show

KeyPointsType = Literal["sift", "orb"]


@img_show("keypoints", cv2.WINDOW_FREERATIO)
def keypoints(img: MatLike, keypoints_type: KeyPointsType) -> MatLike:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model = {"sift": cv2.SIFT_create, "orb": cv2.ORB_create}.get(keypoints_type)(20000)
    keypoints = model.detect(gray, None)
    cv2.drawKeypoints(img, keypoints, img, (0, 255, 0))
    return img


def main():
    keypoints(PopstarAhri, "sift")
    keypoints(PopstarAhri, "orb")


if __name__ == '__main__':
    main()
