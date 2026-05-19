#include <iostream>
#include <string>
#include <vector>

#include "Ahri/Asuka.hpp"

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/ranges.h>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ahri/Ceceilia/utils/loadlibrary.hpp"
#include "Ahri/Asuka/opencv_utils.hpp"
#include "Ahri/Asuka/tensorrt_utils.hpp"
#include "Ahri/Asuka/yolo.hpp"

void yolov5_example() {
    auto model = Ahri::TensorRT::TensorRTModel(std::filesystem::path(ASUKA_ROOT) / "models/yolov5su.onnx");
    cv::Mat img = cv::imread((std::filesystem::path(ASUKA_ROOT) / "images/bus.jpg").string());
    auto preprocess_img = Ahri::YOLO::preprocess(img);
    model.build();
    // std::exit(1);
    model.initialize_engine();
    std::vector<float> outputs = model.inference(preprocess_img);
    auto results = Ahri::YOLO::postprocess_yolov5u(outputs, img.cols, img.rows, 80, 0.25f, 0.7f);
    Ahri::YOLO::plot(img, results);
    IMG_SHOW(img, "TensorRT YOLOv5su Example", cv::WINDOW_FREERATIO);
}

/**
 * @brief AhriNet example is a simple TensorRT plugin example
 */
void ahrinet_example() {
    initLibNvInferPlugins(&Ahri::TensorRT::trtlogger, "");

    // Load plugin DLL
    Ahri::AhriLoadLibrary ahri_plugin_tensorrt{"ahri_plugin_tensorrt"};

    std::vector<float> input_host{
        0.7576, 0.2793, 0.4031, 0.7347, 0.0293,  //
        0.7999, 0.3971, 0.7544, 0.5695, 0.4388,  //
        0.6387, 0.5247, 0.6826, 0.3051, 0.4635,  //
        0.4550, 0.5725, 0.4980, 0.9371, 0.6556,  //
        0.3138, 0.1980, 0.4162, 0.2843, 0.3398,  //
    };
    fmt::println("Input: {}", input_host);

    auto model = Ahri::TensorRT::TensorRTModel(std::filesystem::path(ASUKA_ROOT) / "tensorrt_learn/ahrinet.onnx");
    model.build();
    std::vector<float> result = model.infer(input_host);

    fmt::println("Result: {}", result);
}

int main(int argc, char const* argv[]) {
    fmt::println("CUDA version: {}.{}.{}", CUDA_VERSION / 1000, (CUDA_VERSION % 100) / 10, CUDA_VERSION % 10);
    fmt::println("TensorRT version: {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    ahrinet_example();
    // yolov5_example();

    return 0;
}
