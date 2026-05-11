"""
Reference:
    https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#python-solution-api
"""

import math
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Self, TypeVar

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class Point:
    """Point 2D"""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance(self, p: Self) -> float:
        distance = math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)
        return distance


class Triangle:
    r"""三角形，已知三角形各边长度计算角度

            A
           /\
          /  \
       c /    \ b
        /      \
       /        \
      /          \
     ^^^^^^^^^^^^^^
    B      a       C
    """

    def __init__(self, A: Point, B: Point, C: Point) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.c = A.distance(B)
        self.b = A.distance(C)
        self.a = B.distance(C)

    def angle_A(self):
        cos = (self.c**2 + self.b**2 - self.a**2) / (2 * self.c * self.b)
        angle = round(np.arccos(cos) * 180 / np.pi)
        return angle

    def angle_B(self):
        cos = (self.c**2 + self.a**2 - self.b**2) / (2 * self.c * self.a)
        angle = round(np.arccos(cos) * 180 / np.pi)
        return angle

    def angle_C(self):
        cos = (self.b**2 + self.a**2 - self.c**2) / (2 * self.b * self.a)
        angle = round(np.arccos(cos) * 180 / np.pi)
        return angle


class AbstractCounter(ABC):

    def __init__(self) -> None:
        self.count = 0
        self.visibility_threshold = 0.5
        self.stage = Stage.NONE

    @abstractmethod
    def process(self, *args, **kwargs) -> None:
        raise NotImplementedError


class Stage(IntEnum):
    NONE = 0
    DOWN = 1
    UP = 2


class Situp(AbstractCounter):
    """仰卧起坐"""

    def __init__(self) -> None:
        super().__init__()
        self.situp_curved_threshold = 60  # 弯曲时的角度阈值
        self.situp_spreading_threshold = 120  # 平展时的角度阈值

    def process(self, pose_landmarks, width, height) -> None:
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

        if (
            left_shoulder.visibility > self.visibility_threshold
            and left_hip.visibility > self.visibility_threshold
            # and left_knee.visibility > self.visibility_threshold
        ):
            situp_triangle = Triangle(
                Point(left_shoulder.x * width, left_shoulder.y * height),
                Point(left_hip.x * width, left_hip.y * height),
                Point(left_knee.x * width, left_knee.y * height),
            )
            situp_angle = situp_triangle.angle_B()  # 当前帧腰部弯曲的角度

            if situp_angle > self.situp_spreading_threshold:
                self.stage = Stage.DOWN
            if situp_angle < self.situp_curved_threshold and self.stage == Stage.DOWN:
                self.stage = Stage.UP
                self.count += 1


class Jumprope(AbstractCounter):
    """跳绳"""

    def __init__(self) -> None:
        super().__init__()
        self.last_foot_y = None

    def process(self, pose_landmarks, width, height) -> None:
        """依据左右脚的平均垂直距离判定"""
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        if left_ankle.visibility > self.visibility_threshold and right_ankle.visibility > self.visibility_threshold:
            avg_foot_y = (left_ankle.y * height + right_ankle.y * height) / 2

            if self.last_foot_y is None:
                self.last_foot_y = avg_foot_y
            if avg_foot_y < self.last_foot_y:
                self.stage = Stage.UP
            if avg_foot_y > self.last_foot_y and self.stage == Stage.UP:
                self.stage = Stage.DOWN
                self.count += 1

            self.last_foot_y = avg_foot_y


class Pullup(AbstractCounter):
    """引体向上"""

    def __init__(self) -> None:
        super().__init__()

    def process(self, pose_landmarks, width, height) -> None:
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        if (
            nose.visibility > self.visibility_threshold
            and left_wrist.visibility > self.visibility_threshold
            and right_wrist.visibility > self.visibility_threshold
        ):
            left_hand_y = left_wrist.y * height
            right_hand_y = right_wrist.y * height
            nose_y = nose.y * height

            if left_hand_y < nose_y and right_hand_y < nose_y:
                self.stage = Stage.DOWN

            if left_hand_y > nose_y and right_hand_y > nose_y and self.stage == Stage.DOWN:
                self.stage = Stage.UP
                self.count += 1


class Pushup(AbstractCounter):
    """俯卧撑"""

    def __init__(self) -> None:
        super().__init__()
        self.down_threshold = 80
        self.up_threshold = 120

    def process(self, pose_landmarks, width, height) -> None:
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_elbow_angle = None
        right_elbow_angle = None

        if (
            left_shoulder.visibility > self.visibility_threshold
            and left_elbow.visibility > self.visibility_threshold
            and left_wrist.visibility > self.visibility_threshold
        ):
            left_elbow_triangle = Triangle(
                Point(left_shoulder.x * width, left_shoulder.y * height),
                Point(left_elbow.x * width, left_elbow.y * height),
                Point(left_wrist.x * width, left_wrist.y * height),
            )
            left_elbow_angle = left_elbow_triangle.angle_B()

        if (
            right_shoulder.visibility > self.visibility_threshold
            and right_elbow.visibility > self.visibility_threshold
            and right_wrist.visibility > self.visibility_threshold
        ):
            right_elbow_triangle = Triangle(
                Point(right_shoulder.x * width, right_shoulder.y * height),
                Point(right_elbow.x * width, right_elbow.y * height),
                Point(right_wrist.x * width, right_wrist.y * height),
            )
            right_elbow_angle = right_elbow_triangle.angle_B()

        if left_elbow_angle and right_elbow_angle:
            current_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        elif left_elbow_angle is None:
            current_elbow_angle = right_elbow_angle
        elif right_elbow_angle is None:
            current_elbow_angle = left_elbow_angle

        if current_elbow_angle > self.up_threshold:
            self.stage = Stage.UP
        if current_elbow_angle < self.down_threshold and self.stage == Stage.UP:
            self.stage = Stage.DOWN
            self.count += 1


CounterType = TypeVar("CounterType", bound=AbstractCounter)


def inference_camera[CounterType: AbstractCounter](algorithms: CounterType, mp4_path: str):
    # For webcam input:
    counter: CounterType = algorithms()
    cap = cv2.VideoCapture(mp4_path)
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_FREERATIO)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                counter.process(results.pose_landmarks, width, height)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f"{counter.count}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', image)
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) == 27:
                break
    cap.release()


def main():
    # inference_camera(Situp, r"D:\Download\一分钟仰卧起坐.mp4")
    # inference_camera(Jumprope, r"D:\Download\下载.mp4")
    # inference_camera(Pullup, r"D:\Download\引体向上2.mp4")
    inference_camera(Pushup, r"D:\Download\一分钟60个俯卧撑.mp4")


if __name__ == "__main__":
    main()
