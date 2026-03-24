from pathlib import Path

import librosa


def load_wav(wav_path: Path, sr: int):
    return librosa.core.load(wav_path, sr=sr)[0]
