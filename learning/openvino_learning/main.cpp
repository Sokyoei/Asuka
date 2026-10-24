#include "config.h"

#include <filesystem>

#include <fmt/ranges.h>
#include <opencv2/opencv.hpp>

#include "Asuka/opencv_utils.hpp"
#include "Asuka/openvino_utils.hpp"
#include "Asuka/yolo.hpp"

namespace Asuka::OpenVINO::Samples {
void yolov5_example() {
    auto model_path = std::filesystem::path{ASUKA_ROOT} / "models/yolov5s.xml";
    auto model = Asuka::OpenVINO::OpenVINOModel{model_path};

    // cv::Mat img = cv::imread((std::filesystem::path{SOKYOEI_DATA_DIR} / "Ahri/Popstar Ahri.jpg").string());
    cv::Mat img = cv::imread((std::filesystem::path{ASUKA_ROOT} / "images/zidane.jpg").string());

    cv::Mat preprocess_img = Asuka::YOLO::preprocess(img);
    auto result = model.inference(preprocess_img);
    auto yolo_results = Asuka::YOLO::postprocess(result, img.cols, img.rows);

    Asuka::YOLO::plot(img, yolo_results);
    IMG_SHOW(img, "YOLO Detection Result", cv::WINDOW_FREERATIO)
}
}  // namespace Asuka::OpenVINO::Samples

int main(int argc, char const* argv[]) {
    fmt::println("OpenVINO Version: {}", ov::get_openvino_version());
    Asuka::OpenVINO::Samples::yolov5_example();

    return 0;
}
