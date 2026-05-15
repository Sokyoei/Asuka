"""
测试深度相机视野内两点的真实距离
"""

import math
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)  # streaming流开始
align = rs.align(rs.stream.color)  # 创建对齐对象与color流对齐


def get_aligned_images():
    """获取对齐图像帧与相机参数"""
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


def get_3d_camera_coordinate(point, aligned_depth_frame, depth_intrin):
    """获取随机点三维坐标"""
    distance = aligned_depth_frame.get_distance(*point)  # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, point, distance)
    return distance, camera_coordinate


class RealDistance:

    def __init__(self):
        self.q = deque()

    def get_points(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            if len(self.q) == 2:
                self.q.popleft()
                self.q.append((x, y))
            else:
                self.q.append((x, y))

    def process(self):
        cv2.namedWindow("RealSense", cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback("RealSense", self.get_points)
        while True:
            # 获取对齐图像与相机参数
            _color_intrin, depth_intrin, img_color, _img_depth, aligned_depth_frame = get_aligned_images()

            if len(self.q) == 2:
                (xx1, yy1), (xx2, yy2) = self.q
                p1 = (xx1, yy1)
                dis1, camera_coordinate1 = get_3d_camera_coordinate(p1, aligned_depth_frame, depth_intrin)

                p2 = (xx2, yy2)
                dis2, camera_coordinate2 = get_3d_camera_coordinate(p2, aligned_depth_frame, depth_intrin)

                """显示图像与标注"""
                cv2.circle(img_color, p1, 3, [0, 255, 0], thickness=1)
                cv2.circle(img_color, p1, 6, [0, 255, 0], thickness=1)
                cv2.putText(img_color, f"Dis1: {dis1}  m", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])
                cv2.putText(
                    img_color, f"X1: {camera_coordinate1[0]} m", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0]
                )
                cv2.putText(
                    img_color, f"Y1: {camera_coordinate1[1]} m", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0]
                )
                cv2.putText(
                    img_color, f"Z1: {camera_coordinate1[2]} m", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0]
                )
                cv2.putText(img_color, "1", (xx1 - 5, yy1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])

                cv2.circle(img_color, p2, 3, [255, 0, 255], thickness=1)
                cv2.circle(img_color, p2, 6, [255, 0, 255], thickness=1)
                cv2.putText(img_color, f"Dis2: {dis2}  m", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])
                cv2.putText(
                    img_color, f"X2: {camera_coordinate2[0]} m", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0]
                )
                cv2.putText(
                    img_color, f"Y2: {camera_coordinate2[1]} m", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0]
                )
                cv2.putText(
                    img_color, f"Z2: {camera_coordinate2[2]} m", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0]
                )
                cv2.putText(img_color, "2", (xx2 - 5, yy2 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 255])

                cv2.line(img_color, p1, p2, [0, 255, 255], 1)
                if (
                    camera_coordinate1[0]
                    * camera_coordinate1[1]
                    * camera_coordinate1[2]
                    * camera_coordinate2[0]
                    * camera_coordinate2[1]
                    * camera_coordinate2[2]
                    == 0
                ):
                    cv2.putText(img_color, "Dis1to2: None", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255])
                else:
                    juli = math.sqrt(
                        (camera_coordinate2[0] - camera_coordinate1[0]) ** 2
                        + (camera_coordinate2[1] - camera_coordinate1[1]) ** 2
                        + (camera_coordinate2[2] - camera_coordinate1[2]) ** 2
                    )
                    cv2.putText(img_color, f"Dis1to2: {juli} m", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255])

            cv2.imshow('RealSense', img_color)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


def main():
    r = RealDistance()
    r.process()


if __name__ == "__main__":
    main()
