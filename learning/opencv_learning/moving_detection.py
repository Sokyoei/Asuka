"""
运动物体检测
"""

import itertools
from pathlib import Path
from typing import Literal

import cv2
from cv2.typing import Moments
from matplotlib import pyplot as plt

BackgroundMode = Literal["GMM", "KNN"]


class MovingDetector:

    def __init__(self, mode: BackgroundMode = "KNN"):
        self.model: cv2.BackgroundSubtractorMOG2 | cv2.BackgroundSubtractorKNN = {
            "GMM": cv2.createBackgroundSubtractorMOG2,
            "KNN": cv2.createBackgroundSubtractorKNN,
        }.get(mode)()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.landing_point = []
        self.is_landing_point = False
        self.current_y = 0
        self.x_list = []
        self.y_list = []
        self.ignore_center_points = []
        self.is_get_ignore_center_points = False

    def process(self, video_path: Path):
        cap = cv2.VideoCapture(video_path)
        n = 0
        while True:
            areas = []
            ret, frame = cap.read()
            if not ret:
                break

            if not self.is_get_ignore_center_points:
                self.get_ignore_center_points(frame.shape[:-1])
                self.is_get_ignore_center_points = True

            fmask = self.model.apply(frame)

            fmask = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, self.kernel, iterations=2)

            contours, _ = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # draw max contour
                areas = [cv2.contourArea(contour) for contour in contours]
                max_area_index = areas.index(max(areas))
                max_contour = contours[max_area_index]
                cv2.drawContours(frame, max_contour, -1, (255, 0, 0), 3)
                # draw centroid
                M: Moments = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), 2)

                if [cx, cy] not in self.ignore_center_points:
                    if cy > self.current_y and not self.is_landing_point:
                        self.landing_point = [cx, cy]

                    if cy < self.current_y:
                        self.is_landing_point = True

                    self.x_list.append(n)
                    self.y_list.append(cy)

                    self.current_y = cy

            cv2.putText(frame, f"{self.landing_point}", [0, 25], cv2.FONT_HERSHEY_COMPLEX, 1, [0, 255, 0], 2)

            cv2.imshow('frame', frame)
            # cv2.imshow('fmask', fmask)
            # cv2.imwrite(f"frame_{n}.jpg", frame)

            n += 1

            if cv2.waitKey(10) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        plt.plot(self.x_list, self.y_list, color='r')
        plt.show()

    def get_ignore_center_points(self, frame_size):
        XY_LIMITS = 3
        height, width = frame_size
        center_y, center_x = int(height / 2), int(width / 2)
        center_x_list = [i for i in range(center_x - XY_LIMITS, center_x + XY_LIMITS + 1)]
        center_y_list = [i for i in range(center_y - XY_LIMITS, center_y + XY_LIMITS + 1)]
        self.ignore_center_points = [list(i) for i in itertools.product(center_x_list, center_y_list)]


def main():
    moving = MovingDetector("GMM")
    moving.process(r"D:\Download\实心球视频\【远】第二次投成绩8.6米（左）.mp4")
    # moving.process(r"D:\Download\实心球视频\【远】第三次投踩线违规（左）.mp4")
    # moving.process(r"D:\Download\实心球视频\【远】第四次投成绩8.7米（右）.mp4")
    # moving.process(r"D:\Download\实心球视频\【远】第五次投踩线违规（右）.mp4")
    # moving.process(r"D:\Download\实心球视频\【远】第一次投成绩9.05（左）.mp4")


if __name__ == "__main__":
    main()
