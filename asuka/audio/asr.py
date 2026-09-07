from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path


class ASRType(StrEnum):
    WHISPER = "OpenAI-Whisper"


class BaseASR(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def audio2text(self, audio_path: Path) -> str:
        pass


class OpenAIWhisper(BaseASR):
    import whisper

    def __init__(self):
        super().__init__()
        self.model = self.whisper.load_model("turbo")

    def audio2text(self, audio_path: Path) -> str:
        result = self.model.transcribe(str(audio_path))
        return result["text"]
