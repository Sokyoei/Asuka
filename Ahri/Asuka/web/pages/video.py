import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="video", page_icon="static/assets/favicon.ico", layout="wide")

st.header("capture")
# image_placeholder = st.image([])
# capture = cv2.VideoCapture(0)
# while True:
#     ret, frame = capture.read()

#     if not ret:
#         continue

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

#     image_placeholder.image(frame, channels="BGR")

# captured_image = st.camera_input("Take a picture")
# if captured_image:
#     img = Image.open(captured_image)
#     img = np.array(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
#     st.image(gray_img)

captured_image = st.camera_input("Take a picture")
if captured_image:
    # 将 Streamlit 的图像对象转换为 OpenCV 格式
    img = Image.open(captured_image)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将 RGB 转换为 BGR 格式，因为 OpenCV 使用 BGR 格式

    # 使用 OpenCV 进行图像处理，这里进行边缘检测和模糊处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 100, 200)

    # 将处理后的图像转换回 RGB 格式以便在 Streamlit 中显示
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    st.image(edges)
