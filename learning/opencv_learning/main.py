import cv2
from loguru import logger

from asuka import SOKYOEI_DATA_DIR
from asuka.utils import cv2_utils


def main():
    logger.info("OpenCV version: {}", cv2.__version__)
    logger.info("Number of threads: {}", cv2.getNumThreads())
    logger.info("Number of CPUs: {}", cv2.getNumberOfCPUs())
    logger.info("Current thread number: {}", cv2.getThreadNum())

    # 扩展读取显示图像，支持 CJK 字符。
    ahri = cv2_utils.imread(str(SOKYOEI_DATA_DIR / "Ahri/偶像歌手 阿狸 皇室鸢尾.jpg"))
    if ahri is not None:
        cv2_utils.show_image(ahri, "Ahri")


if __name__ == '__main__':
    main()
