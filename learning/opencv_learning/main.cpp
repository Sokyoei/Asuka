#ifdef _WIN32
#include <Windows.h>
#endif

#include "Asuka.hpp"
#include "Asuka/opencv_utils.hpp"

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    CV_LOG_INFO(nullptr, fmt::format("OpenCV version: {}", cv::getVersionString()));

    Asuka::OpenCV::video_show(true, 0);
    return 0;
}
