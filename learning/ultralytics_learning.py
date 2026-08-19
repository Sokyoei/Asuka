# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -all,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
from typing import cast

import cv2
import matplotlib.pyplot as plt
from ultralytics import ASSETS, YOLO
from ultralytics.engine.results import Results

from Ahri.Asuka.config.config import settings

# %% [markdown]
# [ultralytics](https://github.com/ultralytics/ultralytics)

# %%
model = YOLO(settings.MODELS_DIR / "yolo26n.pt")
results = model(ASSETS / "bus.jpg")
results = cast(Results, results)
# cv2.namedWindow("ultralytics", cv2.WINDOW_FREERATIO)
# cv2.imshow("ultralytics", results[0].plot())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rgb_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.axis("off")
