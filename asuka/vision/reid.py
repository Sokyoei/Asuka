"""
ReID(Re-IDentification)「重识别」
"""

import torch
from cv2.typing import MatLike
from torch import Tensor

from .base import AbstractVisionModel


class TorchReIDExtractor(AbstractVisionModel):
    from torchreid.reid.utils import FeatureExtractor

    def __init__(self, model_name: str = "osnet_ain_x1_0"):
        self.extractor = self.FeatureExtractor(model_name=model_name)

    def inference(self, image: MatLike):
        with torch.no_grad():
            feature: Tensor = self.extractor(image)
        return feature.cpu().numpy().flatten()
