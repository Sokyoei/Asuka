#pragma once
#ifndef COMMON_HPP
#define COMMON_HPP

#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <glib.h>
#include <gst/gst.h>
#include <spdlog/spdlog.h>

inline uint64_t long_to_uint64(long value) noexcept {
    return static_cast<uint64_t>(value & 0xFFFFFFFFFFFFFFFFULL);
}

inline gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer loop) {
    auto* main_loop = static_cast<GMainLoop*>(loop);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            SPDLOG_ERROR("End-of-stream");
            g_main_loop_quit(main_loop);
            break;
        case GST_MESSAGE_WARNING: {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_warning(msg, &err, &debug);
            SPDLOG_WARN("Warning: {}: {}", err->message, debug);
            g_error_free(err);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            SPDLOG_ERROR("ERROR: {}: {}", err->message, debug);
            g_error_free(err);
            g_free(debug);
            g_main_loop_quit(main_loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

class GETFPS {
public:
    explicit GETFPS(const std::string& stream_id) : _stream_id(stream_id), _is_first(true), _frame_count(0) {}

    void update_fps() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto now = std::chrono::steady_clock::now();
        if (_is_first) {
            _start_time = now;
            _is_first = false;
        } else {
            ++_frame_count;
        }
    }

    double get_fps() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - _start_time).count();
        double fps = 0.0;
        if (_frame_count > 0 && elapsed > 0.0) {
            fps = static_cast<double>(_frame_count) / elapsed;
        }
        // 重置计数器
        _frame_count = 0;
        _start_time = now;
        return fps;
    }

    void print_data() const {
        std::lock_guard<std::mutex> lock(_mutex);
        SPDLOG_INFO("steam={} frame_count={} start_time={} ms", _stream_id, _frame_count,
                    std::chrono::duration<double>(_start_time.time_since_epoch()).count());
    }

private:
    std::string _stream_id;
    std::chrono::steady_clock::time_point _start_time;
    bool _is_first;
    uint64_t _frame_count;
    mutable std::mutex _mutex;
};

class PERF_DATA {
public:
    using StreamInfo = std::pair<std::string, std::string>;  // (name, unused)

    PERF_DATA(const std::vector<StreamInfo>& streams) {
        for (const auto& [name, _] : streams) {
            _all_stream_fps.try_emplace(name, name);
        }
    }

    // 周期性调用的回调 (适合作为 GLib 定时器)
    bool perf_print_callback() {
        _perf_dict.clear();
        for (auto& [name, fps_obj] : _all_stream_fps) {
            _perf_dict[name] = fps_obj.get_fps();
        }
        SPDLOG_INFO("[PERF]: {}", _perf_dict);
        return true;  // 返回 true 以在 GLib 定时器中继续
    }

    void update_fps(const std::string& stream_name) {
        auto it = _all_stream_fps.find(stream_name);
        if (it != _all_stream_fps.end()) {
            it->second.update_fps();
        }
    }

private:
    std::map<std::string, GETFPS> _all_stream_fps;
    std::map<std::string, double> _perf_dict;
};

class PlatformInfo {
public:
    PlatformInfo()
        : _wsl_verified(false),
          _is_wsl_system(false),
          _integrated_gpu_verified(false),
          _is_integrated_gpu_system(false),
          _aarch64_verified(false),
          _is_aarch64_platform(false),
          _is_dgx_spark_platform(false) {}

    bool is_wsl() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_wsl_verified) {
            std::ifstream ver_file("/proc/version");
            if (ver_file.is_open()) {
                std::string line;
                std::getline(ver_file, line);
                for (auto& c : line) {
                    c = std::tolower(c);
                }
                if (line.find("microsoft") != std::string::npos) {
                    _is_wsl_system = true;
                }
                _wsl_verified = true;
            } else {
                SPDLOG_ERROR("Opening /proc/version failed.");
            }
        }
        return _is_wsl_system;
    }

    bool is_integrated_gpu() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_integrated_gpu_verified) {
            CUresult res = cuInit(0);
            if (res == CUDA_SUCCESS) {
                int deviceCount = 0;
                res = cuDeviceGetCount(&deviceCount);
                if (res == CUDA_SUCCESS && deviceCount >= 1) {
                    cudaDeviceProp props;
                    cudaError_t err = cudaGetDeviceProperties(&props, 0);
                    if (err == cudaSuccess) {
                        _is_integrated_gpu_system = (props.integrated != 0);
                        SPDLOG_INFO("Is it Integrated GPU? {}.", _is_integrated_gpu_system);
                        _integrated_gpu_verified = true;
                    } else {
                        SPDLOG_ERROR("Getting cuda device property failed: {}.", cudaGetErrorString(err));
                    }
                } else {
                    SPDLOG_ERROR("Getting cuda device count failed: {}.", static_cast<int>(res));
                }
            } else {
                SPDLOG_ERROR("Cuda init failed: {}.", static_cast<int>(res));
            }
        }
        return _is_integrated_gpu_system;
    }

    bool is_platform_aarch64() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_aarch64_verified) {
            struct utsname uname_data;
            if (uname(&uname_data) == 0) {
                if (std::string(uname_data.machine) == "aarch64") {
                    _is_aarch64_platform = true;
                }
            }
            _aarch64_verified = true;
        }
        return _is_aarch64_platform;
    }

    bool is_dgx_spark() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_is_dgx_spark_platform) {
            std::string product_name = read_dmi_field("/sys/class/dmi/id/product_name");
            std::string board_name = read_dmi_field("/sys/class/dmi/id/board_name");
            std::string product_sku = read_dmi_field("/sys/class/dmi/id/product_sku");
            std::string combined = product_name + " " + board_name + " " + product_sku;
            for (auto& c : combined) {
                c = std::tolower(c);
            }
            if (combined.find("dgx spark") != std::string::npos) {
                _is_dgx_spark_platform = true;
            }
        }
        return _is_dgx_spark_platform;
    }

private:
    static std::string read_dmi_field(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return {};
        }
        std::string content;
        std::getline(file, content);
        while (!content.empty() && std::isspace(content.back())) {
            content.pop_back();
        }
        return content;
    }

private:
    bool _wsl_verified;
    bool _is_wsl_system;
    bool _integrated_gpu_verified;
    bool _is_integrated_gpu_system;
    bool _aarch64_verified;
    bool _is_aarch64_platform;
    bool _is_dgx_spark_platform;
    mutable std::mutex _mutex;
};

#endif  // !COMMON_HPP
