import numpy as np
import pyrealsense2 as rs
from loguru import logger

from .base import AbstractCamera


class RealSenseCamera(AbstractCamera):

    def __init__(self, width, height, fps=30):
        super().__init__()
        self.width = width
        self.height = height
        self.pipeline = None
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, 720, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

    def release_workthread(self):
        try:
            self.stopped = True
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.error(f"failed to stop Intel Deep Camera, reason: {e}")
            super().release_workthread()
        except Exception as e:
            logger.error(f"failed to close Intel Deep Camera, reason: {e}")

    def update(self):
        logger.info("RealSense Deep Camera update start")
        self.pipeline = rs.pipeline()

        try:
            # ctx = rs.context()
            # for dev in ctx.query_devices():
            #     dev.hardware_reset()
            # self.pipeline.start(self.config)

            profile = self.pipeline.start(self.config)  # noqa: F841
            # NOTE: 似乎不能加入下面的检查，否则在 x86 小板子上启动时大概率报
            # RuntimeError: Error occured during execution of the processing block! See the log for more info
            # 导致程序不推流，并且线程不中断

            # device = profile.get_device()
            # device.hardware_reset()
        except Exception as e:
            logger.error(f"failed to open Intel Deep Camera, reason: {e}")
            # self.pipeline.stop()
            self.release_workthread()
            return

        # read frames loop
        while not self.stopped:
            # with self.frame_lock:
            # if self.pipeline.poll_for_frames():
            frames = self.pipeline.wait_for_frames()
            frames.keep()

            aligned_frames = self.align.process(frames)

            self.aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()

            self.depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            self.color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

            self.img_color = np.asanyarray(aligned_color_frame.get_data())
            self.img_depth = np.asanyarray(self.aligned_depth_frame.get_data())
            if not self.aligned_depth_frame or not aligned_color_frame:
                break

        # try:
        #     self.pipeline.stop()
        # except Exception as e:
        #     logger.error(f"failed to stop Intel Deep Camera, reason: {e}")

        logger.info("RealSense Deep Camera update end")
