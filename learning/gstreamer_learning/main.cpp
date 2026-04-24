#include <fmt/core.h>
#include <gst/gst.h>

int main(int argc, char const* argv[]) {
    gst_init(nullptr, nullptr);

    fmt::println("Hello, GStreamer! Version: {}", gst_version_string());

    gst_deinit();
    return 0;
}
