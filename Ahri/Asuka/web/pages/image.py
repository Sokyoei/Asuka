import cv2
import streamlit as st
from cv2.typing import MatLike

from Ahri.Asuka import SOKYOEI_DATA_DIR

MORPHOLOGYTPYE = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "blackhat": cv2.MORPH_BLACKHAT,
    "tophat": cv2.MORPH_TOPHAT,
    "gradient": cv2.MORPH_GRADIENT,
}


class Morphology(object):

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def morphology(self, image: MatLike, morphology_type) -> MatLike:
        return cv2.morphologyEx(image, morphology_type, self.kernel)


st.set_page_config(page_title="image", page_icon="static/assets/favicon.ico", layout="wide")

st.header("image")

morphology = Morphology()
morphology_type = st.selectbox("形态学操作", MORPHOLOGYTPYE.keys())
image = cv2.imread(SOKYOEI_DATA_DIR / "Ahri/Popstar Ahri.jpg")
image = morphology.morphology(image, MORPHOLOGYTPYE[morphology_type])
st.image(image, channels="BGR")
