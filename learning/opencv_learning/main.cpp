#include "Ahri/Asuka.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

#include "Ahri/Asuka/opencv_utils.hpp"

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    CV_LOG_INFO(nullptr, fmt::format("OpenCV version: {}", cv::getVersionString()));

    Ahri::OpenCV::video_show(true, 0);
    return 0;
}
