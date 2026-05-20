"""
cascade 级联检测器

HARR(Harr-like)
HOG(Histogram of Oriented Gradient)
LBP(Local Binary Pattern)
"""

from pathlib import Path

import cv2
from cv2.data import haarcascades
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import PopstarAhri, img_show


@img_show("cascade")
def cascade(img: MatLike):
    xml = str(Path(haarcascades) / "haarcascade_eye.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(xml)
    results = cascade.detectMultiScale(gray, 2, 9)
    for box in results:
        cv2.rectangle(img, box, (0, 255, 0), 2)
    return img


def main():
    cascade(PopstarAhri)


if __name__ == '__main__':
    main()
