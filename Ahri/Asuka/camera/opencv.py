import time
from typing import Any, Dict, Optional

import cv2
from loguru import logger

from .base import AbstractCamera


class OpenCVCamera(AbstractCamera):
    """OpenCV摄像头实现"""

    def __init__(self, camera_id: int = 0, resolution: tuple = (640, 480), fps: int = 30, **kwargs):
        """
        初始化OpenCV摄像头

        Args:
            camera_id: 摄像头ID
            resolution: 分辨率 (width, height)
            fps: 帧率
            **kwargs: 传递给基类的参数
        """
        super().__init__(**kwargs)

        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        # OpenCV视频捕获对象
        self.cap = None

        # 更新摄像头信息
        self.camera_info.update(
            {
                "type": "opencv",
                "camera_id": camera_id,
                "resolution": resolution,
                "fps": fps,
                "backend": cv2.getBuildInformation(),
            }
        )

        logger.info(f"[OpenCVCamera] 初始化: 摄像头ID={camera_id}, 分辨率={resolution}, FPS={fps}")

    def __open_camera(self) -> bool:
        """打开OpenCV摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.info(f"[OpenCVCamera] 无法打开摄像头 {self.camera_id}")
                return False

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # 获取实际参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.camera_info.update(
                {
                    "actual_resolution": (actual_width, actual_height),
                    "actual_fps": actual_fps,
                    "fourcc": int(self.cap.get(cv2.CAP_PROP_FOURCC)),
                }
            )

            logger.info("[OpenCVCamera] 摄像头已打开")
            logger.info(f"[OpenCVCamera] 实际分辨率: {actual_width}x{actual_height}")
            logger.info(f"[OpenCVCamera] 实际FPS: {actual_fps}")

            return True

        except Exception as e:
            logger.info(f"[OpenCVCamera] 打开摄像头失败: {e}")
            return False

    def __read_frame(self) -> Optional[Dict[str, Any]]:
        """读取OpenCV摄像头帧"""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            logger.info("[OpenCVCamera] 读取帧失败")
            return None

        # 返回帧数据字典
        return {
            "color": frame.copy(),  # 复制避免后续修改影响
            "timestamp": time.time(),
            "frame_id": self.frame_count,
            "camera_info": self.camera_info.copy(),
        }

    def __close_camera(self):
        """关闭OpenCV摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("[OpenCVCamera] 摄像头已关闭")

    def set_resolution(self, width: int, height: int):
        """动态设置分辨率"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.resolution = (width, height)
            logger.info(f"[OpenCVCamera] 分辨率已设置为: {width}x{height}")

    def set_fps(self, fps: int):
        """动态设置帧率"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
            logger.info(f"[OpenCVCamera] 帧率已设置为: {fps}")
