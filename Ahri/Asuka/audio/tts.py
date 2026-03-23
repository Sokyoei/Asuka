from abc import ABC, abstractmethod


class BaseTTS(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def text2audio(self, text: str):
        pass
