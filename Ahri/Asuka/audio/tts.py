import time
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

import pyaudio
from asgiref.sync import async_to_sync
from loguru import logger


class TTSType(StrEnum):
    EDGETTS = "EdgeTTS"
    COSYVOICE = "CosyVoice"


class BaseTTS(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def text2audio(self, text: str, voice: str | None = None, output_file: Path | None = None, *args, **kwargs):
        pass

    def check_empty_str(self, text: str) -> bool:
        if not text.strip():
            logger.warning("空文本，跳过合成")
            return True
        return False


class EdgeTTS(BaseTTS):
    import edge_tts

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        super().__init__()
        self.voice = voice

    def text2audio(self, text: str, voice: str | None = None, output_file: Path | None = None):
        return async_to_sync(self.text2audio_async)(text, voice, output_file)
        # try:
        #     loop = asyncio.get_running_loop()
        #     future = asyncio.run_coroutine_threadsafe(self.text2audio_async(text, voice, output_file), loop)
        #     return future.result()
        # except RuntimeError:
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     result = loop.run_until_complete(self.text2audio_async(text, voice, output_file))
        #     loop.close()
        #     return result

    async def text2audio_async(self, text: str, voice: str | None = None, output_file: Path | None = None):
        if self.check_empty_str(text):
            return

        await self._generate_audio(text, voice, output_file)

    async def _generate_audio(self, text: str, voice: str | None = None, output_file: Path | None = None):
        voice = voice or self.voice
        communicate = self.edge_tts.Communicate(text, voice)

        # 收集所有音频块
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        # 写入文件
        if output_file:
            with open(output_file, "wb") as f:
                f.write(audio_data)
            logger.info(f"音频文件已保存到 `{output_file}`")
            return


class CosyVoice(BaseTTS):
    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

    def __init__(self, model: str = "cosyvoice-v3-flash", voice: str = "longanling_v3"):
        super().__init__()
        if not self.dashscope.api_key:
            raise RuntimeError("dashscope api_key 未设置")
        self.model = model
        self.voice = voice

    def text2audio(
        self, text: str, model: str | None = None, voice: str | None = None, output_file: Path | None = None
    ):
        if self.check_empty_str(text):
            return

        model = model or self.model
        voice = voice or self.voice
        callback = self.Callback()
        synthesizer = self.SpeechSynthesizer(
            model, voice, format=self.AudioFormat.PCM_22050HZ_MONO_16BIT, callback=callback
        )

        synthesizer.streaming_call(text)
        time.sleep(0.1)
        synthesizer.streaming_complete()

        logger.info(
            f'<CosyVoice> request id: {synthesizer.get_last_request_id()}, 首包延迟为：{synthesizer.get_first_package_delay()}ms'
        )

    class Callback(ResultCallback):
        _player = None
        _stream = None

        def on_open(self):
            logger.info("<CosyVoice> 连接建立")
            self._player = pyaudio.PyAudio()
            self._stream = self._player.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

        def on_complete(self):
            logger.info("<CosyVoice> 语音合成完成，所有合成结果已被接收")

        def on_error(self, message: str):
            logger.error(f"<CosyVoice> 语音合成错误：{message}")

        def on_close(self):
            logger.info("<CosyVoice> 连接关闭")
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._player:
                self._player.terminate()

        def on_event(self, message: str):
            pass

        def on_data(self, data: bytes) -> None:
            logger.info(f"<CosyVoice> 音频长度为：{len(data)}")
            if self._stream:
                self._stream.write(data)
