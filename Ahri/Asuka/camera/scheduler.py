import threading
from concurrent.futures import ThreadPoolExecutor

from .base import AbstractCamera
from .handler import AbstractHandler


class FrameScheduler(threading.Thread):
    """帧调度器，从摄像头队列取帧并分发给处理程序"""

    def __init__(
        self,
        camera: AbstractCamera,
        handlers: list[AbstractHandler] | None = None,
        use_thread_pool: bool = True,
        max_workers: int = 4,
    ):
        super().__init__()
        self.camera = camera
        self.handlers: list[AbstractHandler] = handlers or []
        self.running = False
        self.daemon = True
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if use_thread_pool else None

    def add_handler(self, handler):
        self.handlers.append(handler)

    def run(self):
        self.running = True

        while self.running:
            frame_data = self.camera.get_frame(timeout=0.1)
            if frame_data is None:
                continue
            for handler in self.handlers:
                if self.executor and self.running:
                    self.executor.submit(handler.handle, frame_data)
                else:
                    handler.handle(frame_data)

    def stop(self):
        self.running = False
        self.join(1)

        if self.executor:
            self.executor.shutdown(wait=True)

        for handler in self.handlers:
            handler.close()
