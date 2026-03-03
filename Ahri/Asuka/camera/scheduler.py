import threading
from concurrent.futures import ThreadPoolExecutor

from .base import AbstractCamera
from .handler import AbstractHandler


class FrameScheduler(threading.Thread):
    """帧调度器，从摄像头队列取帧并分发给处理程序"""

    def __init__(self, camera: AbstractCamera, handlers: AbstractHandler = None, use_thread_pool=True, max_workers=4):
        super().__init__()
        self.camera = camera
        self.handlers: list[AbstractHandler] = handlers or []
        self.running = True
        self.daemon = True
        if use_thread_pool:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None

    def add_handler(self, handler):
        self.handlers.append(handler)

    def run(self):
        while self.running:
            frame_data = self.camera.get_frame(timeout=0.1)
            if frame_data is None:
                continue
            for handler in self.handlers:
                if self.executor:
                    self.executor.submit(handler.handle, frame_data)
                else:
                    handler.handle(frame_data)

    def stop(self):
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=True)
