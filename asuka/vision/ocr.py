from enum import IntEnum
from typing import Any

from cv2.typing import MatLike
from loguru import logger

from asuka.utils.cv2_utils import opencv_to_pillow
from asuka.vision.base import AbstractVisionModel


class OCRType(IntEnum):
    paddleocr = 1
    pytesseract = 2
    ddddocr = 3  # https://github.com/sml2h3/ddddocr
    pyocr = 4
    easyocr = 5


class PaddlePaddleOCR(AbstractVisionModel):
    from paddleocr import PaddleOCR, draw_ocr

    def __init__(self):
        super().__init__()
        self.ocr = self.PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

    def inference(self, image: MatLike):
        result = self.ocr.ocr(image, cls=True)
        return result

    def plot(self, data: Any, image: MatLike) -> MatLike:
        for line in data:
            print(line)

        boxes = [elements[0] for elements in data[0]]
        txts = [elements[1][0] for elements in data[0]]
        scores = [elements[1][1] for elements in data[0]]

        im_show = self.draw_ocr(image, boxes, txts, scores)
        return im_show


class PyTesseractOCR(AbstractVisionModel):
    import pytesseract

    def __init__(self):
        super().__init__()
        self.config = " ".join(
            [
                # "--psm",
                # "7",
                "-c",
                "tessedit_char_whitelist=0123456789",  # 设置白名单
                "tessedit_write_images=true",  # 设置输出日志
            ]
        )

    def inference(self, image: MatLike):
        image = opencv_to_pillow(image)
        # using image_to_* API
        text = self.pytesseract.image_to_boxes(image, config=self.config)
        return text


class DDDDOCR(AbstractVisionModel):
    import ddddocr

    def __init__(self):
        super().__init__()
        self.ocr = self.ddddocr.DdddOcr()
        self.ocr.set_ranges(0)

    def inference(self, image: MatLike):
        image = opencv_to_pillow(image)
        result = self.ocr.classification(image, probability=False)
        return result
        # s = ""
        # for i in result['probability']:
        #     s += result['charsets'][i.index(max(i))]
        # print(s)


class PyOCR(AbstractVisionModel):
    import pyocr
    import pyocr.builders

    def __init__(self):
        super().__init__()
        tools = self.pyocr.get_available_tools()
        if len(tools) == 0:
            logger.error("No OCR tool found")
            raise FileNotFoundError("No OCR tool found")
        self.ocr = tools[0]

    def inference(self, image: MatLike):
        image = opencv_to_pillow(image)
        text = self.ocr.image_to_string(image, lang='chi_sim', builder=self.pyocr.builders.TextBuilder())
        return text


class EasyOCR(AbstractVisionModel):
    import easyocr

    def __init__(self):
        super().__init__()
        self.ocr = self.easyocr.Reader(["en"])

    def inference(self, image: MatLike):
        image = opencv_to_pillow(image)
        results = self.ocr.readtext(image, detail=0, allowlist="0123456789")
        return results


class OCR:

    def __init__(self, ocr_type: OCRType):
        self.ocr_type = ocr_type

        self.models: dict[OCRType, type[AbstractVisionModel]] = {
            OCRType.paddleocr: PaddlePaddleOCR,
            OCRType.pytesseract: PyTesseractOCR,
            OCRType.ddddocr: DDDDOCR,
            OCRType.pyocr: PyOCR,
            OCRType.easyocr: EasyOCR,
        }
        self.model = self.models[self.ocr_type]()

    def inference(self, image: MatLike):
        return self.model.inference(image)
