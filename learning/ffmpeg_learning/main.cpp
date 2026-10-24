/**
 * @file ffmpeg_learn.cpp
 * @date 2024/09/10
 * @author Sokyoei
 *
 *
 */

#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <fmt/core.h>

#include "Asuka/ffmpeg_utils.hpp"

int main(int argc, char const* argv[]) {
    fmt::println("avcodec version: {}", LIBAVCODEC_IDENT);
    fmt::println("avformat version: {}", LIBAVFORMAT_IDENT);
    fmt::println("avdevice version: {}", LIBAVDEVICE_IDENT);
    fmt::println("avutil version: {}", LIBAVUTIL_IDENT);
    fmt::println("swscale version: {}", LIBSWSCALE_IDENT);
    fmt::println("swresample version: {}", LIBSWRESAMPLE_IDENT);

    Asuka::FFmpeg::init_ffmpeg();

    Asuka::FFmpeg::Media media;
    if (!media.open("rtsp://127.0.0.1:8554/1")) {
        fmt::println("open failed");
        return -1;
    }

    Asuka::FFmpeg::deinit_ffmpeg();
    return 0;
}
