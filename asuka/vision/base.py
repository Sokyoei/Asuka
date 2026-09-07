from abc import ABC, abstractmethod
from typing import Any

from cv2.typing import MatLike


class AbstractVisionModel(ABC):

    @abstractmethod
    def inference(self, image: MatLike):
        raise NotImplementedError

    def infer(self, image: MatLike):
        return self.inference(image)

    def plot(self, data: Any, image: MatLike) -> MatLike:
        return image
