/**
 * @file ffmpeg_utils.hpp
 * @date 2025/01/13
 * @author Sokyoei
 * FFmpeg Utils
 *
 */

#pragma once
#ifndef AHRI_FFMPEG_UTILS_HPP
#define AHRI_FFMPEG_UTILS_HPP

#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/version.h>
// #include <libpostproc/postprocess.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace Ahri::FFmpeg {
inline void init_ffmpeg() {
#if LIBAVCODEC_VERSION_MAJOR < 59
    av_register_all();
    avcodec_register_all();
    avformat_register_all();
#else  // FFmpeg 4.x ^^^ / vvv FFmpeg 5.x later
    avformat_network_init();
#endif
}

inline void deinit_ffmpeg() {
#if LIBAVCODEC_VERSION_MAJOR < 59

#else  // FFmpeg 4.x ^^^ / vvv FFmpeg 5.x later
    avformat_network_deinit();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Log
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macro
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef _MSC_VER
static char av_error[AV_ERROR_MAX_STRING_SIZE] = {0};
#ifdef av_err2str
#undef av_err2str
#endif
#define av_err2str(errnum) av_make_error_string(av_error, AV_ERROR_MAX_STRING_SIZE, errnum)
#endif

#define AV_CHECK(retcode, message)                                       \
    do {                                                                 \
        if (retcode != 0)                                                \
            av_log(nullptr, AV_LOG_ERROR, message, av_err2str(retcode)); \
        return EXIT_FAILURE                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Media
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Media {
public:
    Media() {
        _format_context = avformat_alloc_context();
        _packet = av_packet_alloc();
    }

    ~Media() {
        if (_codec_context) {
            avcodec_free_context(&_codec_context);
        }
        if (_format_context) {
            avformat_close_input(&_format_context);
        }
        if (_packet) {
            av_packet_free(&_packet);
        }
        if (_frame) {
            av_frame_free(&_frame);
        }
        if (_sws_context) {
            sws_freeContext(_sws_context);
        }
    }

    bool open(const char* url) {
        // 打开输入文件
        if (avformat_open_input(&_format_context, url, nullptr, nullptr) != 0) {
            return false;
        };

        // 查找流信息
        if (avformat_find_stream_info(_format_context, nullptr) < 0) {
            return false;
        }

        // 查找视频流
        for (uint8_t i = 0; i < _format_context->nb_streams; i++) {
            if (_format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                _video_stream_index = i;
                break;
            }
        }

        if (_video_stream_index == -1) {
            return false;
        }

        // 获取解码器
        const AVCodec* codec = avcodec_find_decoder(_format_context->streams[_video_stream_index]->codecpar->codec_id);
        if (!codec) {
            return false;
        }

        // 分配解码器上下文
        _codec_context = avcodec_alloc_context3(codec);
        if (!_codec_context) {
            return false;
        }

        // 复制编解码器参数到解码器上下文
        if (avcodec_parameters_to_context(_codec_context, _format_context->streams[_video_stream_index]->codecpar) <
            0) {
            return false;
        }

        // 打开解码器
        if (avcodec_open2(_codec_context, codec, nullptr) < 0) {
            return false;
        }

        // 分配帧
        _frame = av_frame_alloc();
        if (!_frame) {
            return false;
        }

        return true;
    }

    // 获取视频流索引
    int get_video_stream_index() const { return _video_stream_index; }

    // 获取格式上下文
    AVFormatContext* get_format_context() const { return _format_context; }

    // 获取视频编解码器参数
    AVCodecParameters* get_video_codecpar() const {
        if (_video_stream_index >= 0 && _video_stream_index < (int)_format_context->nb_streams) {
            return _format_context->streams[_video_stream_index]->codecpar;
        }
        return nullptr;
    }

    // 读取下一帧
    bool read_frame() {
        while (av_read_frame(_format_context, _packet) >= 0) {
            if (_packet->stream_index == _video_stream_index) {
                // 解码视频帧
                int ret = avcodec_send_packet(_codec_context, _packet);
                if (ret < 0) {
                    av_packet_unref(_packet);
                    continue;
                }

                ret = avcodec_receive_frame(_codec_context, _frame);
                if (ret == 0) {
                    // 成功解码一帧
                    av_packet_unref(_packet);
                    return true;
                }
            }
            av_packet_unref(_packet);
        }
        return false;
    }

    // 获取当前帧
    AVFrame* get_frame() const { return _frame; }

    // 获取视频宽度
    int get_width() const {
        if (_codec_context) {
            return _codec_context->width;
        }
        return 0;
    }

    // 获取视频高度
    int get_height() const {
        if (_codec_context) {
            return _codec_context->height;
        }
        return 0;
    }

    // 获取视频帧率
    double get_fps() const {
        if (_format_context && _video_stream_index >= 0) {
            AVRational rational = _format_context->streams[_video_stream_index]->avg_frame_rate;
            if (rational.den != 0) {
                return static_cast<double>(rational.num) / rational.den;
            }
        }
        return 0.0;
    }

    // 转换帧格式为RGB24
    uint8_t* convert_to_rgb24() {
        if (!_frame) {
            return nullptr;
        }

        // 分配RGB帧
        if (!_rgb_frame) {
            _rgb_frame = av_frame_alloc();
            _rgb_frame->format = AV_PIX_FMT_RGB24;
            _rgb_frame->width = _codec_context->width;
            _rgb_frame->height = _codec_context->height;
            av_frame_get_buffer(_rgb_frame, 32);
        }

        // 初始化SWS上下文
        if (!_sws_context) {
            _sws_context = sws_getContext(_codec_context->width, _codec_context->height, _codec_context->pix_fmt,
                                          _codec_context->width, _codec_context->height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                                          nullptr, nullptr, nullptr);
        }

        // 执行转换
        sws_scale(_sws_context, (const uint8_t* const*)_frame->data, _frame->linesize, 0, _codec_context->height,
                  _rgb_frame->data, _rgb_frame->linesize);

        return _rgb_frame->data[0];
    }

private:
    AVFormatContext* _format_context = nullptr;
    AVCodecContext* _codec_context = nullptr;
    AVPacket* _packet = nullptr;
    AVFrame* _frame = nullptr;
    AVFrame* _rgb_frame = nullptr;
    SwsContext* _sws_context = nullptr;
    int _video_stream_index = -1;
};
}  // namespace Ahri::FFmpeg

#endif  // !AHRI_FFMPEG_UTILS_HPP
