import cv2

from Ahri.Asuka.camera.handler import SaveHandler
from Ahri.Asuka.camera.opencv import OpenCVCamera
from Ahri.Asuka.camera.scheduler import FrameScheduler
from Ahri.Asuka.config.config import settings


def test_opencv():
    camera = OpenCVCamera(camera_id=0)
    camera.start()
    camera.wait_until_ready()

    scheduler = FrameScheduler(camera)
    scheduler.add_handler(SaveHandler(settings.VIDEOS_DIR / "test_opencv.mp4"))
    scheduler.start()

    i = 0
    cv2.namedWindow("OpenCV Display", cv2.WINDOW_FREERATIO)

    try:
        while True:
            frame = camera.get_frame(timeout=2.0)
            i += 1
            if frame is not None:
                print(f"获取到第 {i} 帧")
                cv2.imshow('OpenCV Display', frame['color'])
                cv2.waitKey(1)
            else:
                print(f"第 {i} 次获取超时")
    except KeyboardInterrupt:
        scheduler.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_opencv()
