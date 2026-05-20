#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "Ahri/Asuka.hpp"
#include "Ahri/Asuka/opencv_utils.hpp"

namespace Ahri {
void FLD(cv::Mat& img) {
    cv::Mat img_gray{img.size(), CV_8UC1};
    cv::Mat plot_img = img.clone();
    std::vector<cv::Vec4f> lines;

    auto fld = cv::ximgproc::createFastLineDetector();
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    fld->detect(img_gray, lines);
    fld->drawSegments(plot_img, lines);
    // gray ?
    IMG_SHOW(plot_img, "FLD", cv::WINDOW_FREERATIO)
}

void LSD(cv::Mat& img) {
    cv::Mat img_gray{img.size(), CV_8UC1};
    cv::Mat plot_img = img.clone();
    std::vector<cv::Vec4f> lines;

    auto lsd = cv::createLineSegmentDetector();
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    lsd->detect(img_gray, lines);
    lsd->drawSegments(plot_img, lines);
    IMG_SHOW(plot_img, "LSD", cv::WINDOW_FREERATIO)
}

void EDlines(cv::Mat& img) {
    cv::Mat img_gray{img.size(), CV_8UC1};
    cv::Mat plot_img = img.clone();
    std::vector<cv::Vec4f> lines;

    auto ed = cv::ximgproc::createEdgeDrawing();

    // EDParams
    ed->params.MinPathLength = 50;
    ed->params.PFmode = false;
    ed->params.MinLineLength = 10;
    ed->params.NFAValidation = true;

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    ed->detectEdges(img_gray);
    ed->detectLines(lines);
    for (auto&& points : lines) {
        cv::line(plot_img, cv::Point(points[0], points[1]), cv::Point(points[2], points[3]), cv::Scalar(0, 255, 0));
    }
    IMG_SHOW(plot_img, "EDlines", cv::WINDOW_FREERATIO)
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto img_path = std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Popstar Ahri.jpg";
    auto img = cv::imread(img_path.string(), cv::IMREAD_COLOR);

    Ahri::FLD(img);
    Ahri::LSD(img);
    Ahri::EDlines(img);

    return 0;
}
