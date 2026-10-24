#pragma once
#ifndef ASUKA_REALSENSE2_UTILS_HPP
#define ASUKA_REALSENSE2_UTILS_HPP

#include <fmt/core.h>
#include <librealsense2/rs.hpp>

namespace Asuka::RealSense2 {
inline int check_device() {
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
        fmt::println("Couldn't find RealSense device.");
    } else {
        fmt::println("Find {} RealSense devices.", devices.size());
        for (int i = 0; i < devices.size(); i++) {
            rs2::device device = devices[i];
            try {
                fmt::println("\tDevice: {}", i + 1);
                fmt::println("\tSN: {}", device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
                fmt::println("\tName: {}", device.get_info(RS2_CAMERA_INFO_NAME));
            } catch (const rs2::error& e) {
                fmt::println("Get device error: {}", e.what());
                return EXIT_FAILURE;
            } catch (const std::exception& e) {
                fmt::println("Unknow error: {}", e.what());
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }
}
}  // namespace Asuka::RealSense2

#endif  // !ASUKA_REALSENSE2_UTILS_HPP
