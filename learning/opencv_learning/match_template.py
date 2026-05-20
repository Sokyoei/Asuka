"""
模板匹配
"""

import cv2

from Ahri.Asuka import ASUKA_ROOT
from Ahri.Asuka.utils.cv2_utils import img_show


@img_show("match template")
def match_template():
    img = cv2.imread(str(ASUKA_ROOT / r"data\Ahri\Popstar Ahri.jpg"))
    templ = cv2.imread(r"D:\Antares\Ahri\Popstar Ahri\Popstar Ahri-1024x1024.png")
    tw, th, _tz = templ.shape

    matchs = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(matchs)
    cv2.rectangle(img, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 2, cv2.LINE_AA)
    return img


def main():
    match_template()


if __name__ == "__main__":
    main()
