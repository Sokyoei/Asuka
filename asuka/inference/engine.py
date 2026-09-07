from enum import IntEnum


class InferenceType(IntEnum):
    TensorRT = 0
    OpenVINO = 1
    ONNXRuntime = 2
    OpenCV_DNN = 3


class InferenceEngine:

    def __init__(self, inference_type: InferenceType):
        self.inference_type = inference_type
