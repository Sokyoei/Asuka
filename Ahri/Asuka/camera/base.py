import threading
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import Any, Dict, Optional

from loguru import logger


class AbstractCamera(threading.Thread, ABC):
    """抽象摄像头基类，使用队列提高效率"""

    def __init__(self, max_queue_size: int = 10, frame_timeout: float = 0.1, drop_when_full: bool = True):
        super().__init__()
        self.max_queue_size = max_queue_size
        self.frame_timeout = frame_timeout
        self.drop_when_full = drop_when_full

        # frame variable
        self.frame_queue = Queue(maxsize=self.max_queue_size)
        # self.frame: MatLike | NoneType = None

        # thread variable
        self.daemon = True
        self.name = f"{self.__class__.__name__}Thread"

        # control variable
        self.running = True
        self.frame_count = 0
        self.drop_count = 0

        self.cond = threading.Condition()
        self.stats_lock = threading.Lock()
        self.ready_event = threading.Event()

        self.stats = {"fps": 0, "queue_size": 0, "frame_drops": 0, "last_update": time.time()}
        self.camera_info = {}

        logger.info(f"<{self.name}> 初始化完成，队列大小: {max_queue_size}")

    @abstractmethod
    def __open_camera(self) -> bool:
        """打开摄像头"""
        pass

    @abstractmethod
    def __read_frame(self):
        """读取一帧"""
        pass

    @abstractmethod
    def __close_camera(self):
        """关闭摄像头"""
        pass

    def run(self):
        """主循环，读取帧并放入队列"""
        logger.info(f"{self.name} 开始运行")

        if not self.__open_camera():
            logger.error(f"{self.name} 无法打开摄像头")
            return

        self.ready_event.set()

        fps_counter = 0
        last_fps_time = time.time()

        try:
            while self.running:
                # 读取一帧
                frame_data = self.__read_frame()

                if frame_data is None:
                    time.sleep(0.001)  # 短暂休眠避免 CPU 占用过高
                    continue

                self.frame_count += 1
                fps_counter += 1

                # 更新 FPS 统计
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    with self.stats_lock:
                        self.stats["fps"] = fps_counter
                        self.stats["queue_size"] = self.frame_queue.qsize()
                        self.stats["frame_drops"] = self.drop_count
                    fps_counter = 0
                    last_fps_time = current_time

                # 将帧放入队列
                if self.frame_queue.full():
                    if self.drop_when_full:
                        # 丢弃最老的帧
                        try:
                            self.frame_queue.get_nowait()
                            self.drop_count += 1
                        except Empty:
                            pass
                    else:
                        # 等待队列有空位
                        pass

                # 放入新帧（非阻塞，如果队列已满则丢弃）
                try:
                    self.frame_queue.put_nowait(frame_data)
                except Exception:
                    self.drop_count += 1

        except Exception as e:
            logger.error(f"{self.name} 错误！原因：{e}")
        finally:
            self.__close_camera()
            logger.info(f"{self.name} 已停止，总帧数: {self.frame_count}")

    def get_frame(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        从队列获取一帧

        Args:
            timeout: 超时时间，None表示使用默认超时

        Returns:
            帧数据字典，包含'color'、'depth'等字段，失败返回None
        """
        if timeout is None:
            timeout = self.frame_timeout

        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            logger.warning(f"[{self.name}] 获取帧超时")
            return None

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的帧（清空队列并返回最后一帧）

        Returns:
            最新的帧数据
        """
        latest_frame = None

        # 清空队列，只保留最新的帧
        while not self.frame_queue.empty():
            try:
                latest_frame = self.frame_queue.get_nowait()
            except Empty:
                break

        return latest_frame

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """
        等待摄像头准备就绪

        Args:
            timeout: 超时时间

        Returns:
            是否就绪
        """
        return self.ready_event.wait(timeout)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取摄像头统计信息

        Returns:
            统计信息字典
        """
        with self.stats_lock:
            stats = self.stats.copy()
            stats.update(
                {
                    "total_frames": self.frame_count,
                    "total_drops": self.drop_count,
                    "camera_info": self.camera_info.copy(),
                }
            )
        return stats

    def clear_queue(self):
        """清空帧队列"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

    def stop(self):
        """停止摄像头线程"""
        self.running = False
        self.clear_queue()
        self.join(timeout=2.0)

    def __del__(self):
        """析构函数，确保资源释放"""
        if self.running:
            self.stop()
