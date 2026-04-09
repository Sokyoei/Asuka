"""
OpenVINO Utils
"""

from __future__ import annotations

import os
from abc import ABC

import cv2
import numpy as np
import openvino as ov
from cv2.typing import MatLike
from openvino.preprocess import PrePostProcessor

if int(ov.__version__.split(".")[0]) >= 2026:
    from openvino.utils.data_helpers.wrappers import OVDict
else:
    from openvino.runtime.utils.data_helpers.wrappers import OVDict


class OpenVINOModel(ABC):

    def __init__(self, model_path: os.PathLike | str) -> None:
        super.__init__()
        self.model_path = model_path
        self.device_name = "CPU"

        # initialize runtime engine
        self.core = ov.Core()

        self.model = self.core.read_model(self.model_path)
        if self.model.inputs != 1:
            raise RuntimeError("")
        if self.model.outputs != 1:
            raise RuntimeError("")
        ppp = PrePostProcessor(self.model)

        image = cv2.imread("")
        input_tensor = np.expand_dims(image, 0)

        ppp.input().tensor().set_shape(input_tensor.shape).set_element_type(ov.Type.u8).set_layout(ov.Layout('NHWC'))
        ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
        ppp.input().model().set_layout(ov.Layout('NCHW'))
        ppp.output().tensor().set_element_type(ov.Type.f32)
        self.model = ppp.build()

        self.compiled_model = self.core.compile_model(self.model, self.device_name)

    def inference(self, image: MatLike) -> OVDict:
        input_tensor = np.expand_dims(image, 0)
        results = self.compiled_model.infer_new_request({0: input_tensor})
        return results

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError
