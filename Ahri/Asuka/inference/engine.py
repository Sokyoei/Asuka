from enum import IntEnum


class InferenceType(IntEnum):
    tensorrt = 0
    openvino = 1
    onnxruntime = 2


class InferenceEngine:

    def __init__(self, inference_type: InferenceType):
        self.inference_type = inference_type
