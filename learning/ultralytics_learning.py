# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
import cv2
from ultralytics import ASSETS, YOLO
from ultralytics.engine.results import Results

from Ahri.Asuka.config.config import settings

# %% [markdown]
# [ultralytics](https://github.com/ultralytics/ultralytics)

# %%
model = YOLO(settings.MODELS_DIR / "yolo26n.pt")
results: Results = model(ASSETS / "bus.jpg")
cv2.namedWindow("ultralytics", cv2.WINDOW_FREERATIO)
cv2.imshow("ultralytics", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
