/**
 * @file opencv_utils.hpp
 * @date 2024/07/17
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef ASUKA_OPENCV_UTILS_HPP
#define ASUKA_OPENCV_UTILS_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/std.h>
#include <opencv2/opencv.hpp>

#include "Asuka.hpp"
#include "Ceceilia/utils/logger.hpp"
#include "Ceceilia/utils/size_fmt.hpp"

namespace Asuka::OpenCV {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define IMG_SHOW(img, winname, flag) \
    cv::namedWindow(winname, flag);  \
    cv::imshow(winname, img);        \
    cv::waitKey();                   \
    cv::destroyAllWindows();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename... Args>
inline void video_show(bool screen, Args&&... args) {
    cv::VideoCapture capture{args...};
    cv::VideoWriter writer;

    // capture properties
    double fps = capture.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    if (screen) {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        // fmt library remove fmt::localtime in version 12.0.0 ??
#ifdef FMT_VERSION >= 120000
        std::tm tm = *std::localtime(&t);
#else
        std::tm tm = fmt::localtime(t);
#endif
        std::string filename = fmt::format("screen_{:%Y%m%d_%H%M%S}.mp4", tm);

        writer.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

        if (!writer.isOpened()) {
            AHRI_LOGGER_ERROR("Can not create {}.", filename);
            screen = false;
        }
    }

    while (true) {
        cv::Mat frame;

        bool ret = capture.read(frame);
        if (!ret) {
            break;
        }

        if (screen) {
            writer.write(frame);
        }

        cv::putText(frame, std::to_string(fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),
                    1, cv::LINE_AA);
        cv::imshow("Camera", frame);
        auto c = cv::waitKey(1);
        if (c == 27) {
            break;
        }
    }

    capture.release();
    writer.release();
}

inline std::string depth2str(int depth) {
    switch (depth) {
            // clang-format off
        case CV_8U:  return "CV_8U";
        case CV_8S:  return "CV_8S";
        case CV_16U: return "CV_16U";
        case CV_16S: return "CV_16S";
        case CV_32S: return "CV_32S";
        case CV_32F: return "CV_32F";
        case CV_64F: return "CV_64F";
        case CV_16F: return "CV_16F";
        default:     return "UNKNOW";
            // clang-format on
    }
}

inline void print_mat_info(const cv::Mat& mat) {
    if (mat.empty()) {
        AHRI_LOGGER_WARN("Empty Mat!");
        return;
    }

    AHRI_LOGGER_DEBUG("Mat type: {}C{}", depth2str(CV_MAT_DEPTH(mat.type())), mat.channels());
    AHRI_LOGGER_DEBUG("Mat continuous: {}", mat.isContinuous());
    AHRI_LOGGER_DEBUG("Mat elements: {}x{}={}", mat.cols, mat.rows, mat.total());
    size_t bytes = mat.total() * mat.elemSize();
    AHRI_LOGGER_DEBUG("Mat memory: {}({})", bytes, Ceceilia::size_fmt(bytes));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Classes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class OpenCVDNNModel {
public:
    /**
     * @param model_path ONNX model file path
     */
    OpenCVDNNModel(std::filesystem::path model_path) : _model_path(model_path) { load_model(); }

    ~OpenCVDNNModel() {}

    void load_model() {
        if (_model_loaded) {
            return;
        }

        _net = cv::dnn::readNetFromONNX(_model_path.string());

        auto layer_names = _net.getLayerNames();
        AHRI_LOGGER_DEBUG("Model has {} layers", layer_names.size());

        _model_loaded = true;
    }

    /**
     * @brief 使用 cv::Mat 输入进行推理
     * @param input_mat 输入图像 RGB
     * @return 推理结果向量
     */
    cv::Mat inference(const cv::Mat& input_mat) {
        _net.setInput(input_mat);
        cv::Mat output = _net.forward();
        return output;
    }

private:
    std::filesystem::path _model_path;
    cv::dnn::Net _net;
    bool _model_loaded = false;
};

using Model = OpenCVDNNModel;
}  // namespace Asuka::OpenCV

#endif  // !ASUKA_OPENCV_UTILS_HPP
