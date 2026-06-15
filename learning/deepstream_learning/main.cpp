#include <fmt/core.h>
#include <gst/gst.h>
#include <nvds_version.h>

int main(int argc, char* argv[]) {
    fmt::println("Hello, GStreamer! Version: {}", gst_version_string());
    nvds_version_print();

    gst_init(&argc, &argv);
    nvds_dependencies_version_print();

    return 0;
}
