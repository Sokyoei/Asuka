#include <filesystem>

#include "Ahri/Asuka.hpp"

#ifdef _WIN32
#include <Windows.h>
#else defined(__linux__)
#include <unistd.h>
#endif

#include "Ahri/Asuka/onnx_utils.hpp"
#include "Ahri/Asuka/opencv_utils.hpp"
#include "Ahri/Asuka/yolo.hpp"

namespace Ahri::ONNX {
void yolov5su_example() {
    try {
        auto model = Ahri::ONNX::ONNXRuntimeModel(std::filesystem::path(ASUKA_ROOT) / "models/yolov5su.onnx");
        cv::Mat img = cv::imread((std::filesystem::path(ASUKA_ROOT) / "images/bus.jpg").string());
        auto preprocess_img = Ahri::YOLO::preprocess(img);
        std::vector<float> outputs = model.inference(preprocess_img);
        auto results = Ahri::YOLO::postprocess_yolov5u(outputs, img.cols, img.rows, 80, 0.25f, 0.7f);
        Ahri::YOLO::plot(img, results);
        IMG_SHOW(img, "ONNXRuntime YOLOV5SU Example", cv::WINDOW_FREERATIO);
    } catch (const Ort::Exception& e) {
        AHRI_LOGGER_ERROR("Ort::Exception: {}", e.what());
    }
}
}  // namespace Ahri::ONNX

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // auto model = Ahri::ONNX::ONNXRuntimeModel(std::filesystem::path(VISION_ROOT) / "models/VGG.onnx");
    // cv::Mat image{cv::Size(224, 224), CV_8UC3, cv::Scalar(0.5f)};
    // cv::Mat image_float;
    // image.convertTo(image_float, CV_32FC3, 1.0 / 255.0);
    // auto result = model.inference(image_float);

    Ahri::ONNX::yolov5su_example();

    return 0;
}
