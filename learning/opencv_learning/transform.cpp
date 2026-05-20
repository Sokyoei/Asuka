#include <filesystem>

#include <opencv2/opencv.hpp>

#include "Ahri/Asuka.hpp"
#include "Ahri/Asuka/opencv_utils.hpp"

namespace Ahri {
void affine(cv::Mat& img) {
    cv::Mat plot_img{img.size(), CV_8UC3};
    // cv::Mat M = cv::getAffineTransform();

    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    double angle = 30.0;
    double scale = 1.0;
    cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);

    // cv::Rect2f bbox = cv::RotatedRect(center, img.size(), angle).boundingRect2f();
    // M.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    // M.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    cv::warpAffine(img, plot_img, M, plot_img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    IMG_SHOW(plot_img, "affine", cv::WINDOW_FREERATIO)
}

void perspective(cv::Mat& img) {
    cv::Mat plot_img{img.size(), CV_8UC3};

    const cv::Point2f src_points[4]{
        cv::Point2f(0, 0),
        cv::Point2f(img.cols, 0),
        cv::Point2f(0, img.rows),
        cv::Point2f(img.cols, img.rows),
    };
    const cv::Point2f dst_points[4]{
        cv::Point2f(0, 0),
        cv::Point2f(img.cols * 0.9f, img.rows * 0.1f),
        cv::Point2f(img.cols * 0.1f, img.rows * 0.9f),
        cv::Point2f(img.cols * 0.75f, img.rows * 0.75f),
    };
    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    cv::warpPerspective(img, plot_img, M, plot_img.size());
    IMG_SHOW(plot_img, "perspective", cv::WINDOW_FREERATIO)
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto img_path = std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Popstar Ahri.jpg";
    auto img = cv::imread(img_path.string(), cv::IMREAD_COLOR);

    Ahri::affine(img);
    Ahri::perspective(img);
    return 0;
}
