import os
import subprocess
from enum import IntEnum
from pathlib import Path


class KeyFrameType(IntEnum):
    FFmpeg = 1
    Katna = 2


class KeyFrame(object):

    def __init__(
        self, keyframe_type: KeyFrameType = KeyFrameType.FFmpeg, keyframe_extract_dir: str = "keyframe_extract"
    ):
        self.keyframe_type = keyframe_type
        self.keyframe_extract_path = Path(keyframe_extract_dir)
        self.keyframe_extract_path.mkdir(exist_ok=True, parents=True)
        self.keyframe_extract_funcs = {KeyFrameType.Katna: self._katna, KeyFrameType.FFmpeg: self._ffmpeg}

    def extract_keyframe(self, video_path: str | os.PathLike):
        self.keyframe_extract_funcs[self.keyframe_type](video_path)

    def _ffmpeg(self, video_path: str | os.PathLike):
        # fmt: off
        cmd = [
            "ffmpeg",
            "-i",           video_path,
            "-vf",          r"select=eq(pict_type\,I)",
            "-vsync",       "vfr",
            "-qscale:v",    "2",
            "-f",           "image2",
            f"{self.keyframe_extract_path}/%08d.jpg",
        ]
        # fmt: on
        try:
            result = subprocess.run(cmd)
            result.check_returncode()
            return True
        except subprocess.CalledProcessError:
            return False

    def _katna(self, video_path: str | os.PathLike):
        """
        https://github.com/keplerlab/Katna
        https://blog.csdn.net/qq_15969343/article/details/124157138
        """

        from Katna.video import Video
        from Katna.writer import KeyFrameDiskWriter

        try:
            vd = Video()
            no_of_frames_to_returned = 12
            disk_writer = KeyFrameDiskWriter(location=str(self.keyframe_extract_path))
            vd.extract_video_keyframes(no_of_frames=no_of_frames_to_returned, file_path=video_path, writer=disk_writer)
        except Exception:
            return False
        return True
