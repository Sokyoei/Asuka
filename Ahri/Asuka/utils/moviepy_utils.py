from pathlib import Path

import moviepy


def webm2mp4(webm_path: Path, mp4_path: Path):
    """
    Warnings:
        似乎无法处理 Chrome 录制的 VP9 编码 + 透明通道的特殊 WebM
    """
    with moviepy.VideoFileClip(webm_path) as clip:
        clip.write_videofile(
            mp4_path,
            codec="libx264",  # 视频编码 H.264
            audio_codec="aac",  # 音频编码 AAC
            bitrate="3000k",  # 码率 3000kbps
            fps=clip.fps,
            preset="fast",
        )
