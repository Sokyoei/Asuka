from pathlib import Path

import av


def webm2mp4(webm_path: Path, mp4_path: Path) -> None:
    input_container = av.open(webm_path)
    output_container = av.open(mp4_path, "w", format="mp4")

    # 视频流配置
    video_stream = input_container.streams.video[0]
    out_video = output_container.add_stream("libx264", rate=video_stream.average_rate)
    out_video.width = video_stream.width
    out_video.height = video_stream.height
    out_video.pix_fmt = "yuv420p"
    out_video.bit_rate = 3000_0000
    out_video.options = {"preset": "fast"}

    # 音频流配置
    audio_stream = input_container.streams.audio[0]
    out_audio = output_container.add_stream("aac", rate=audio_stream.rate)
    out_audio.bit_rate = 12_8000

    # 逐帧解码、转码、封装
    for packet in input_container.demux():
        # 解码数据包
        for frame in packet.decode():
            if packet.stream.type == "video":
                for pkt in out_video.encode(frame):
                    output_container.mux(pkt)
            elif packet.stream.type == "audio":
                for pkt in out_audio.encode(frame):
                    output_container.mux(pkt)

    # 刷线缓存中剩余的帧
    for pkt in out_video.encode(None):
        output_container.mux(pkt)
    for pkt in out_audio.encode(None):
        output_container.mux(pkt)

    input_container.close()
    output_container.close()
