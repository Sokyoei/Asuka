#include <fmt/core.h>
#include <gst/gst.h>

#ifdef _WIN32
#include <Windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    GstElement* pipeline = nullptr;                       // 管道
    GstBus* bus = nullptr;                                // 总线
    GstMessage* message = nullptr;                        // 消息
    GError* error = nullptr;                              // 错误
    GstStateChangeReturn ret = GST_STATE_CHANGE_FAILURE;  // 状态变化返回值

    // 初始化 GStreamer
    gst_init(&argc, &argv);
    fmt::println("Hello, GStreamer! Version: {}", gst_version_string());

    // Vcpkg GStreamer 插件
    const char* vcpkg_gstreamer_plugin_path = R"(D:\vcpkg\installed\x64-windows\plugins\gstreamer)";
    g_setenv("GST_PLUGIN_PATH", vcpkg_gstreamer_plugin_path, TRUE);
    gst_registry_scan_path(gst_registry_get(), vcpkg_gstreamer_plugin_path);

    // 创建管道
    pipeline = gst_parse_launch("videotestsrc ! autovideosink", &error);
    if (error != nullptr) {
        fmt::println("Error: {}", error->message);
        goto end;
    }

    // 开始播放
    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fmt::println("Error: Failed to start pipeline");
        goto end;
    }

    // 等待错误或窗口关闭
    bus = gst_element_get_bus(pipeline);
    message = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

end:
    // 释放资源
    if (message != nullptr) {
        gst_message_unref(message);
    }
    if (bus != nullptr) {
        gst_object_unref(bus);
    }
    if (pipeline != nullptr) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
    if (error != nullptr) {
        g_error_free(error);
    }

    gst_deinit();
    return 0;
}
