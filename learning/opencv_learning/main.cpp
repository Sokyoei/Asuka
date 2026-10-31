#include "config.h"

#include <cmath>
#include <filesystem>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <fmt/base.h>
#include <opencv2/opencv.hpp>

#include "Asuka.hpp"
#include "Asuka/opencv_utils.hpp"
#include "Ceceilia/utils/logger.hpp"

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    CV_LOG_INFO(nullptr, fmt::format("OpenCV version: {}", cv::getVersionString()));

    Asuka::OpenCV::video_show(true, 0);

    auto file_path = (std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Popstar Ahri ASU.jpg").string();
    cv::Mat kda_ahri = cv::imread(file_path);
    if (kda_ahri.empty()) {
        AHRI_LOGGER_ERROR("Image not found.");
        return EXIT_FAILURE;
    }
    IMG_SHOW(kda_ahri, "Popstar Ahri ASU", cv::WINDOW_FREERATIO)
    Asuka::OpenCV::print_mat_info(kda_ahri);

    return 0;
}
