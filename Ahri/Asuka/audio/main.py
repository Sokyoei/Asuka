from pathlib import Path

import librosa
from pydub.audio_segment import AudioSegment
from pydub.playback import play

from Ahri.Asuka import SOKYOEI_DATA_DIR


def load_wav(wav_path: Path, sr: int):
    return librosa.core.load(wav_path, sr=sr)[0]


def play_audio(audio_path: Path):
    audio = AudioSegment.from_file(SOKYOEI_DATA_DIR / "Ahri/KDA_POP_STARS.mp3")
    play(audio)
