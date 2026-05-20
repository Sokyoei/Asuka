/**
 * @file morphological.cpp
 * @date 2024/07/17
 * @author Sokyoei
 *
 *
 */

#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "Ahri/Asuka.hpp"
#include "Ahri/Asuka/opencv_utils.hpp"

namespace Ahri {
void erode(cv::Mat& img) {
    cv::Mat dst;
    auto kernel = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(5, 5));
    cv::erode(img, dst, kernel);
    IMG_SHOW(dst, "erode", cv::WINDOW_FREERATIO)
}

void dilate(cv::Mat& img) {
    cv::Mat dst;
    auto kernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(5, 5));
    cv::dilate(img, dst, kernel);
    IMG_SHOW(dst, "dilate", cv::WINDOW_FREERATIO)
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto img_path = std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Popstar Ahri.jpg";
    auto img = cv::imread(img_path.string());

    Ahri::erode(img);
    Ahri::dilate(img);

    cv::destroyAllWindows();
    return 0;
}
