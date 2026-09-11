# .SYNOPSIS
# 单个摄像头推送多个 RTSP 视频流
# .NOTES
# 获取 DShow 需要的摄像头名称
# ```shell
# ffmpeg -list_devices true -f dshow -i dummy
# ```

$CAMERA_NAME = "Intel(R) RealSense(TM) Depth Camera 455  RGB"

$STREAM_COUNT = 50
$TEE_STR = (1..$STREAM_COUNT | ForEach-Object { "[f=rtsp:rtsp_transport=tcp]rtsp://127.0.0.1:8554/$_" }) -join "|"
ffmpeg.exe -f dshow `
    -rtbufsize 256M `
    -video_size 640x480 `
    -i video=$CAMERA_NAME `
    -map 0:v `
    -c:v libx264 `
    -preset ultrafast `
    -tune zerolatency `
    -b:v 2000k `
    -pix_fmt yuv420p `
    -f tee "$TEE_STR"
