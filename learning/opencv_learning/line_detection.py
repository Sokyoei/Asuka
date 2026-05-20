"""
直线检测
"""

import cv2
from cv2.typing import MatLike

from Ahri.Asuka.utils.cv2_utils import GREEN, img_show


@img_show("FLD")
def FLD(img: MatLike):
    fld = cv2.ximgproc.createFastLineDetector()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = fld.detect(img_gray)
    plot_img = fld.drawSegments(img, lines, linecolor=GREEN)
    return plot_img


@img_show("LSD")
def LSD(img: MatLike):
    lsd = cv2.createLineSegmentDetector()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines, _width, _prec, _nfs = lsd.detect(img_gray)
    img = lsd.drawSegments(img, lines)
    return img


@img_show("EDlines")
def EDlines(img: MatLike):
    ed = cv2.ximgproc.createEdgeDrawing()

    # EDParams
    edparams = cv2.ximgproc.EdgeDrawing.Params()
    edparams.MinPathLength = 50
    edparams.PFmode = False
    edparams.MinLineLength = 10
    edparams.NFAValidation = True
    ed.setParams(edparams)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ed.detectEdges(img_gray)
    lines = ed.detectLines()
    lines = lines.astype(int)
    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img, (x1, y1), (x2, y2), GREEN, 2, cv2.LINE_AA)
    return img


def main():
    img = cv2.imread(r"D:\Andromeda\Sokyoei\Asuka\images\wx_20241120152605.jpg")
    FLD(img)
    LSD(img)
    EDlines(img)


if __name__ == "__main__":
    main()
