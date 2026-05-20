/**
 * @file keypoints.cpp
 * @date 2024/07/17
 * @author Sokyoei
 *
 *
 */

#include <filesystem>
#include <iostream>
#include <vector>

#include "Ahri/Asuka.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "Ahri/Asuka/opencv_color.hpp"

namespace Ahri {
enum class KeyPointsType {
    SIFT,
    ORB,
};

void keypoints(cv::Mat& img) {
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    auto model = cv::SIFT::create(20000);
    std::vector<cv::KeyPoint> keypoints;
    model->detect(img, keypoints);
    cv::drawKeypoints(img, keypoints, img, GREEN);
    cv::namedWindow("keypoints", cv::WINDOW_FREERATIO);
    cv::imshow("keypoints", img);
    cv::waitKey();
    cv::destroyAllWindows();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto img_path = std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Popstar Ahri.jpg";
    auto img = cv::imread(img_path.string());
    Ahri::keypoints(img);
    return 0;
}
