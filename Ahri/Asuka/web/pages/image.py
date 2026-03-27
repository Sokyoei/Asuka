import streamlit as st
from cv2.typing import MatLike
from PIL import Image

from Ahri.Asuka.utils.cv2_utils import (
    FILTER_TYPES,
    MORPHOLOGY_SHAPES,
    MORPHOLOGY_TYPES,
    PopstarAhri,
    filter_image,
    morphology,
    pillow_to_opencv,
)

st.set_page_config(page_title="image", page_icon="static/assets/favicon.ico", layout="wide")

st.header("image")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = []


def morphology_operator(image: MatLike, index: int) -> MatLike:
    with st.expander(f"形态学操作 #{index}", True):
        morph_type = st.selectbox("形态学操作", MORPHOLOGY_TYPES.keys(), key=f"morphology_type_{index}")
        morph_shape = st.selectbox("形态学形状", MORPHOLOGY_SHAPES.keys(), key=f"morphology_shape_{index}")
        morph_count = st.slider("形态学迭代次数", 1, 10, key=f"morphology_count_{index}")
        image = morphology(image, morph_type, morph_shape, morph_count)
        return image


def filter_operator(image: MatLike, index: int) -> MatLike:
    with st.expander(f"滤波操作 #{index}", True):
        filter_type = st.selectbox("滤波操作", FILTER_TYPES.keys(), key=f"filter_type_{index}")
        image = filter_image(image, filter_type)
        return image


image = PopstarAhri.copy()
upload_image = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
if upload_image:
    image = pillow_to_opencv(Image.open(upload_image))

with st.sidebar:
    for index, operator in enumerate(st.session_state.pipeline):
        image = operator(image, index)

    st.button("添加算子", on_click=lambda: st.session_state.pipeline.append(morphology_operator))

st.image(image, channels="BGR")
