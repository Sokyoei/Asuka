from abc import ABC, abstractmethod

import cv2
from cv2.typing import MatLike


class AbstractHandler(ABC):

    @abstractmethod
    def handle(self, frame_data: MatLike):
        pass


class DisplayHandler(AbstractHandler):

    def __init__(self, window_name: str = "Display"):
        self.window_name = window_name

    def handle(self, frame_data: MatLike):
        cv2.imshow(self.window_name, frame_data["color"])
        cv2.waitKey(0.01)


class SaveHandler(AbstractHandler):

    def __init__(self, save_path: str, resolution: tuple[int, int] = (640, 480), fps: int = 30):
        self.save_path = save_path
        self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)

    def handle(self, frame_data: MatLike):
        self.writer.write(frame_data["color"])

    def __del__(self):
        self.writer.release()


class PushHandler(AbstractHandler):

    def __init__(self, rtmp_url: str):
        pass

    def handle(self, frame_data: MatLike):
        pass
