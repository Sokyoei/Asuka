"""
使用 FFmpeg 将视频文件转换为图片序列
"""

import subprocess
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-file", help="输入视频文件路径")
    parser.add_argument("-o", "--output-dir", help="输出文件夹路径")
    parser.add_argument("-vf", "--video-frame", help="每秒提取帧数", default="5")
    args = parser.parse_args()

    try:
        print("开始视频转换...")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # fmt: off
        cmd = [
            "ffmpeg",
            "-i",           f"{args.input_file}",
            "-f",           "image2",
            "-vf",          f"fps={args.video_frame}",
            "-qscale:v",    "2",
            f"{args.output_dir}/%04d.jpg",
        ]
        # fmt: on
        result = subprocess.run(cmd, shell=True)
        result.check_returncode()
        print("视频转换成功！")
    except subprocess.CalledProcessError:
        print("视频转换失败！")


if __name__ == "__main__":
    main()
