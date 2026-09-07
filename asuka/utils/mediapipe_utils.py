import cv2
import mediapipe as mp
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class MediaPipeHand:

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def inference(
        self, image: MatLike, return_source_image: bool = False
    ) -> tuple[MatLike, NDArray] | tuple[MatLike, NDArray, MatLike]:
        if return_source_image:
            src_image = image.copy()

        height, width, _ = image.shape
        points = np.array([])
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        if results.multi_hand_landmarks:
            points = np.array(
                [
                    [
                        [landmark.x * width, landmark.y * height, landmark.visibility]
                        for _, landmark in enumerate(hand_landmarks.landmark)
                    ]
                    for hand_landmarks in results.multi_hand_landmarks
                ]
            )
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if return_source_image:
            return image, points, src_image
        else:
            return image, points

    def __del__(self):
        self.hands.close()
