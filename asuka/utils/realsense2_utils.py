import numpy as np
import pyrealsense2 as rs
from loguru import logger


########################################################################################################################
# functions
########################################################################################################################
def check_device():
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            logger.info(
                f'Found device: {d.get_info(rs.camera_info.name)}, SN: {d.get_info(rs.camera_info.serial_number)}'
            )
        return True
    else:
        logger.error("No Intel Device connected")
        return False


class RealSenseDeepCamera:

    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        colorizer = rs.colorizer()
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        return color_image, depthx_image, colorizer_depth

    def __del__(self):
        self.pipeline.stop()
