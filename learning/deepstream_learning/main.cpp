#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif  // !SPDLOG_ACTIVE_LEVEL

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <fmt/core.h>
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gstnvdsmeta.h>
#include <nvds_version.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "common.hpp"

constexpr auto VIDEO_SOURCE_URI = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264";
// constexpr const std::vector<std::pair<std::string, std::string>> VIDEO_SOURCE_URIS = {
//     {"1", "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"},
//     {"2", "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"},
//     {"3", "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"},
// };
constexpr auto PGIE_CONFIG_FILE = "/workspace/DeepStream-Yolo/config_infer_primary_yolo26.txt";
// nvtracker, 也适用于 YOLO
constexpr auto TRACKER_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
constexpr auto TRACKER_CONFIG_FILE =
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
constexpr auto SUPPORT_URI_PREFIXES = {"rtsp://", "http://", "https://", "file://"};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Logger
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @see https://github.com/Sokyoei/Ceceilia/blob/master/include/Ahri/Ceceilia/utils/logger_utils.hpp
 */
#define COLOR_GREEN "\033[32m"
#define COLOR_CYAN "\033[36m"
#define COLOR_RESET "\033[0m"

inline void init_logging(const std::string& file_path, spdlog::level::level_enum log_level = spdlog::level::trace) {
    // sinks
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_logger = std::make_shared<spdlog::sinks::daily_file_sink_mt>(file_path, 0, 0);
    std::string console_pattern = fmt::format("{}[%Y-%m-%d %H:%M:%S.%e]{}{}[%s:%#]{}%^[%l]: %v%$", COLOR_GREEN,
                                              COLOR_RESET, COLOR_CYAN, COLOR_RESET);
    console->set_pattern(console_pattern);
    file_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%s:%#]%^[%l]: %v%$");

    std::initializer_list<spdlog::sink_ptr> sinks{console, file_logger};
    auto loggers = std::make_shared<spdlog::logger>("deepstream_logger", sinks);
    loggers->set_level(log_level);
    spdlog::register_logger(loggers);
    spdlog::set_default_logger(loggers);
}

#define CHECK_FACTORY(name, element_ptr)                    \
    if (element_ptr == nullptr) {                           \
        SPDLOG_ERROR("Failed to create element: {}", name); \
        return nullptr;                                     \
    }

#define CHECK_FACTORYS(elements)                  \
    for (auto&& [name, element_ptr] : elements) { \
        CHECK_FACTORY(name, element_ptr);         \
    }

GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    GstBuffer* buffer = GST_BUFFER(info->data);
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    // 遍历每一帧
    for (auto l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
        // 遍历每个检测物
        for (auto l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
        }
    }

    return GstPadProbeReturn::GST_PAD_PROBE_OK;
}

class DeepStreamPipeline {
public:
    DeepStreamPipeline(bool display = false) : _display(display) {}

    ~DeepStreamPipeline() {
        if (_pipeline) {
            gst_element_set_state(_pipeline, GstState::GST_STATE_NULL);
            gst_object_unref(GST_OBJECT(_pipeline));
            _pipeline = nullptr;
        }
        g_source_remove(_bus_watch_id);
        if (_loop) {
            g_main_loop_unref(_loop);
            _loop = nullptr;
        }
        _elements.clear();
    }

    void start() {
        _loop = g_main_loop_new(nullptr, false);
        _pipeline = build_pipeline();
        if (!_pipeline) {
            SPDLOG_CRITICAL("Failed to build pipeline.");
            std::exit(EXIT_FAILURE);
        }

        _bus = gst_pipeline_get_bus(GST_PIPELINE(_pipeline));
        _bus_watch_id = gst_bus_add_watch(_bus, bus_call, _loop);
        gst_object_unref(_bus);

        gst_element_set_state(_pipeline, GstState::GST_STATE_PLAYING);
        g_main_loop_run(_loop);
    }

private:
    GstElement* build_pipeline(std::string pipeline_name = "pipeline") {
        _pipeline = gst_pipeline_new(pipeline_name.c_str());

        // Config: {factoryname: name}
        std::vector<std::pair<std::string, std::string>> elements_config = {
            {       "filesrc",      "source"},
            {     "h264parse",      "parser"},
            { "nvv4l2decoder",     "decoder"},
            {   "nvstreammux", "streammuxer"},
            {       "nvinfer",        "pgie"},
            {     "nvtracker",     "tracker"},
            {"nvvideoconvert",   "converter"},
            {       "nvdsosd",        "plot"},
        };
        add_osd_sink(elements_config);

        // Create elements
        for (auto&& [factoryname, name] : elements_config) {
            auto element_ptr = gst_element_factory_make(factoryname.c_str(), name.c_str());
            _elements[name] = element_ptr;
        }
        CHECK_FACTORYS(_elements);

        // Set properties
        g_object_set(_elements["source"], "location", VIDEO_SOURCE_URI, nullptr);
        g_object_set(_elements["parser"], "config-interval", -1, nullptr);
        g_object_set(_elements["streammuxer"], "width", 1920, nullptr);
        g_object_set(_elements["streammuxer"], "height", 1080, nullptr);
        g_object_set(_elements["streammuxer"], "batch-size", 1, nullptr);
        g_object_set(_elements["streammuxer"], "batched-push-timeout", 40000, nullptr);
        g_object_set(_elements["pgie"], "config-file-path", PGIE_CONFIG_FILE, nullptr);
        g_object_set(_elements["tracker"], "ll-lib-file", TRACKER_LIB_FILE, nullptr);
        g_object_set(_elements["tracker"], "ll-config-file", TRACKER_CONFIG_FILE, nullptr);
        g_object_set(_elements["tracker"], "gpu_id", 0, nullptr);

        // Add elements to bin
        for (auto&& [_, element] : _elements) {
            gst_bin_add(GST_BIN(_pipeline), element);
        }

        // Link elements
        if (!gst_element_link_many(_elements["source"], _elements["parser"], _elements["decoder"], nullptr)) {
            SPDLOG_ERROR("Failed to link source -> parse -> deocder");
            return nullptr;
        }

        if (!gst_element_link_pads(_elements["decoder"], "src", _elements["streammuxer"], "sink_0")) {
            SPDLOG_ERROR("Failed to link decoder -> streammuxer");
            return nullptr;
        }
        // GstPad* decoder_sink_pad = gst_element_get_static_pad(_elements["decoder"], "src");
        // if (!decoder_sink_pad) {
        //     SPDLOG_ERROR("Failed to get decoder sink pad.");
        //     return nullptr;
        // }
        // GstPad* streammuxer_sink_pad = gst_element_request_pad_simple(_elements["streammuxer"], "sink_0");
        // if (!streammuxer_sink_pad) {
        //     SPDLOG_ERROR("Failed to request streammuxer sink pad.");
        //     gst_object_unref(decoder_sink_pad);
        //     return nullptr;
        // }
        // if (gst_pad_link(decoder_sink_pad, streammuxer_sink_pad) != GST_PAD_LINK_OK) {
        //     SPDLOG_ERROR("Failed to link decoder -> streammuxer");
        //     gst_object_unref(decoder_sink_pad);
        //     gst_object_unref(streammuxer_sink_pad);
        //     return nullptr;
        // }
        // gst_object_unref(decoder_sink_pad);
        // gst_object_unref(streammuxer_sink_pad);
        // SPDLOG_INFO("Successfully linked decoder -> streammuxer.");

        if (!gst_element_link_many(_elements["streammuxer"], _elements["pgie"], _elements["tracker"],
                                   _elements["converter"], _elements["plot"], _elements["show"], nullptr)) {
            SPDLOG_ERROR("Failed to link backend elements");
            return nullptr;
        }

        GstPad* osd_sink_pad = gst_element_get_static_pad(_elements["plot"], "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GstPadProbeType::GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe,
                              nullptr, nullptr);
            gst_object_unref(osd_sink_pad);
        }

        SPDLOG_INFO("DeepStream pipeline built successfully.");
        return _pipeline;
    }

    void add_osd_sink(std::vector<std::pair<std::string, std::string>>& elements_config) {
        if (_display) {
            // CUDA 设备属性
            int current_device = -1;
            cudaGetDevice(&current_device);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, current_device);

            if (prop.integrated) {
                elements_config.push_back({"nv3dsink", "show"});
            } else {
#ifdef __aarch64__
                elements_config.push_back({"nv3dsink", "show"});
#else
                elements_config.push_back({"nveglglessink", "show"});
#endif
            }
        } else {
            elements_config.push_back({"fakesink", "show"});
        }
    }

private:
    bool _display = false;
    GMainLoop* _loop = nullptr;
    GstBus* _bus = nullptr;
    guint _bus_watch_id;
    GstElement* _pipeline = nullptr;
    std::map<std::string, GstElement*> _elements = {};  // {name: GstElement}
};

int main(int argc, char* argv[]) {
    init_logging("/workspace/logs/deepstream.log");

    // Version print
    fmt::println("Hello, GStreamer! Version: {}", gst_version_string());

    gst_init(&argc, &argv);
    nvds_version_print();
    nvds_dependencies_version_print();

    try {
        DeepStreamPipeline pipeline{true};
        pipeline.start();
        SPDLOG_INFO("DeepStream pipeline started.");
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception caught: {}", e.what());
    }

    gst_deinit();
    spdlog::drop_all();

    return 0;
}
