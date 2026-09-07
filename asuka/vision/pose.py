from enum import IntEnum

from cv2.typing import MatLike

from asuka.vision.base import AbstractVisionModel


class PoseType(IntEnum):
    rtmlib = 1  # https://github.com/Tau-J/rtmlib
    mediapipe = 2
    openpose = 3


class RTMLibPose(AbstractVisionModel):
    from rtmlib import Wholebody, draw_skeleton

    def __init__(self):
        super().__init__()
        device = 'cpu'  # cpu, cuda, mps
        backend = 'onnxruntime'  # opencv, onnxruntime, openvino
        openpose_skeleton = False  # True for openpose-style, False for mmpose-style

        self.wholebody = self.Wholebody(
            to_openpose=openpose_skeleton,
            mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
            backend=backend,
            device=device,
        )

    def inference(self, image: MatLike):
        keypoints, scores = self.wholebody(image)
        return keypoints, scores

    def plot(self, data, image):
        keypoints, scores = data
        return self.draw_skeleton(image, keypoints, scores, kpt_thr=0.5)


class Pose:

    def __init__(self, pose_type: PoseType):
        self.pose_type = pose_type

    def inference(self, image: MatLike):
        pass
