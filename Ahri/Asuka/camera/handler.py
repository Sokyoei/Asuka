from abc import ABC, abstractmethod
from pathlib import Path

import cv2
from cv2.typing import MatLike
from loguru import logger


class AbstractHandler(ABC):

    @abstractmethod
    def handle(self, frame_data: MatLike):
        pass

    @abstractmethod
    def close(self):
        pass


########################################################################################################################
# Handlers
########################################################################################################################
# WARNING: cv2.imshow()/cv2.waitKey() 不能在子线程中使用，否则导致显示卡死
# class DisplayHandler(AbstractHandler):

#     def __init__(self, window_name: str = "Display"):
#         self.window_name = window_name
#         cv2.namedWindow(self.window_name, cv2.WINDOW_FREERATIO)

#     def handle(self, frame_data: MatLike):
#         cv2.imshow(self.window_name, frame_data["color"])
#         cv2.waitKey(1)

#     def close(self):
#         cv2.destroyWindow(self.window_name)


class SaveHandler(AbstractHandler):

    def __init__(self, save_path: Path, resolution: tuple[int, int] = (640, 480), fps: int = 30):
        self.save_path = save_path
        self.resolution = resolution
        self.fps = fps

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)

    def handle(self, frame_data: MatLike):
        self.writer.write(frame_data["color"])

    def close(self):
        if self.writer:
            self.writer.release()
            logger.info(f"视频文件保存至 {self.save_path}")


class RTMPPushHandler(AbstractHandler):
    import av

    def __init__(self, rtmp_url: str, resolution: tuple[int, int] = (640, 480), fps: int = 30):
        # 打开 RTMP 输出流
        self.output_container = self.av.open(rtmp_url, 'w', format='flv')
        # 创建视频流，指定编码器为 H.264
        self.stream = self.output_container.add_stream('h264', rate=fps)
        self.stream.width, self.stream.height = resolution
        self.stream.pix_fmt = 'yuv420p'

        # 调整编码器参数以减少延迟
        self.stream.options = {
            'preset': 'ultrafast',  # 使用超快预设，减少编码时间
            'tune': 'zerolatency',  # 调整编码器以实现零延迟
            # 'crf': '23',  # 恒定速率因子，控制视频质量
            "maxrate": "1000k",
            "bufsize": "1000k",
        }

    def handle(self, frame_data: MatLike):
        frame_rgb = cv2.cvtColor(frame_data["color"], cv2.COLOR_BGR2RGB)

        # 创建 PyAV 帧
        av_frame = self.av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

        # 将帧编码为数据包
        for packet in self.stream.encode(av_frame):
            if packet:
                # 将数据包写入输出流
                self.output_container.mux(packet)

    def __del__(self):
        self.output_container.close()


class WebRTCPushHandler(AbstractHandler):

    def __init__(self):
        pass

    def handle(self, frame_data: MatLike):
        pass
