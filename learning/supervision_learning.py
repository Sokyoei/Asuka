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
import numpy as np
import supervision as sv
from numpy.typing import NDArray
from ultralytics import ASSETS, YOLO

from Ahri.Asuka.config.config import settings

# %% [markdown]
# [supervision](https://github.com/roboflow/supervision)

# %%
model = YOLO(settings.MODELS_DIR / "yolo26n.pt")
image = cv2.imread(str(ASSETS / "bus.jpg"))
results = model(image)
detections = sv.Detections.from_ultralytics(results[0])

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
image = cast(NDArray[np.uint8], image)
annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.axis("off")
