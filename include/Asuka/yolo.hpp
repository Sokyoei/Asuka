/**
 * @file yolo.hpp
 * @date 2024/07/15
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef ASUKA_YOLO_HPP
#define ASUKA_YOLO_HPP

#include <cmath>

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Asuka/opencv_color.hpp"
#include "Ceceilia/utils/logger.hpp"

namespace Asuka::YOLO {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int input_width = 640;
constexpr int input_height = 640;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Result
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Result {
    int class_id;
    float score;
    cv::Rect2f normalized_box;  // 添加归一化后的box
    cv::Rect mapped_box;        // 添加映射到原始图像上的box
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YOLO Version
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class YOLOVersion { YOLOV3, YOLOV5, YOLOV7, YOLOV8, YOLOV9, YOLOV10, YOLO_WORLD, YOLOX };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Labels
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief coco80 类别
 */
std::vector<std::string> coco80{
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush",
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Sigmoid 激活函数，将输入值映射到 (0, 1) 区间
 * @param x 输入值（通常为 float 类型）
 * @return 映射后的值（范围：0 < result < 1）
 */
inline float sigmoid(float x) {
    if (x > 20.0f) {
        return 1.0f;
    } else if (x < -20.0f) {
        return 0.0f;
    }
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Non-Maximum Suppression
 *
 * @param results
 * @param conf_threshold
 * @param iou_threshold
 * @return std::vector<Result>
 */
inline std::vector<Result> nms(std::vector<Result> results, float conf_threshold, float iou_threshold) {
    std::vector<Result> nms_result;
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.score > b.score; });

    for (size_t i = 0; i < results.size(); ++i) {
        // First check confidence threshold
        if (results[i].score < conf_threshold) {
            continue;
        }

        bool keep = true;
        for (size_t j = 0; j < nms_result.size(); ++j) {
            // Skip if either box is invalid
            if (results[i].mapped_box.width <= 0 || results[i].mapped_box.height <= 0 ||
                nms_result[j].mapped_box.width <= 0 || nms_result[j].mapped_box.height <= 0) {
                continue;
            }

            // Only apply NMS to same class objects
            if (results[i].class_id != nms_result[j].class_id) {
                continue;
            }

            cv::Rect intersection = results[i].mapped_box & nms_result[j].mapped_box;

            // Calculate IoU
            float intersection_area = static_cast<float>(intersection.area());
            if (intersection_area <= 0) {
                continue;
            }

            float union_area =
                static_cast<float>(results[i].mapped_box.area() + nms_result[j].mapped_box.area() - intersection_area);
            if (union_area <= 0) {
                continue;
            }

            float iou = intersection_area / union_area;

            if (iou > iou_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            nms_result.push_back(results[i]);
        }
    }

    return nms_result;
}

/**
 * @brief 预处理，将图像由 BGR 转换为 RGB，缩放到 640x640，并将像素值归一化到 (0, 1) 区间
 * @param image 输入图像
 * @return cv::Mat
 */
inline cv::Mat preprocess(cv::Mat& image) {
    return cv::dnn::blobFromImage(image, 1.0f / 255.0f, cv::Size(640, 640), cv::Scalar(), true, false, CV_32F);
}

inline std::vector<Result> postprocess(std::vector<float> output,
                                       int original_width,
                                       int original_height,
                                       int num_classes = 80,
                                       float conf_threshold = 0.25f,
                                       float nms_threshold = 0.7f) {
    const int elements_per_box = 5 + num_classes;
    std::vector<Result> detections;

    // 遍历所有检测框
    for (size_t i = 0; i < output.size(); i += elements_per_box) {
        float conf = output[i + 4];
        if (conf < conf_threshold) {
            continue;
        }

        // 提取 xywh 坐标
        float x = output[i];
        float y = output[i + 1];
        float w = output[i + 2];
        float h = output[i + 3];

        // 转换xywh到xyxy
        float x1 = x - w / 2.0f;  // 左上角x
        float y1 = y - h / 2.0f;  // 左上角y
        float x2 = x + w / 2.0f;  // 右下角x
        float y2 = y + h / 2.0f;  // 右下角y

        // 找到最大类别分数的索引
        int class_id = 0;
        float max_class_score = output[i + 5];  // 第一个类别的分数
        for (int c = 1; c < num_classes; ++c) {
            float class_score = output[i + 5 + c];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                class_id = c;
            }
        }

        float score = conf * max_class_score;

        cv::Rect2f normalized_box{x1, y1, w, h};
        cv::Rect mapped_box{
            static_cast<int>(x1 / input_width * original_width), static_cast<int>(y1 / input_height * original_height),
            static_cast<int>(w / input_width * original_width), static_cast<int>(h / input_height * original_height)};

        detections.push_back({class_id, score, normalized_box, mapped_box});
    }

    return nms(detections, conf_threshold, nms_threshold);
}

/**
 * @brief 适用于 ultralytics 框架训练出的 YOLOv5
 * @param output
 * @param original_width
 * @param original_height
 * @param num_classes
 * @param conf_threshold
 * @param nms_threshold
 * @param enable_nms
 * @return std::vector<Result>
 */
inline std::vector<Result> postprocess_yolov5u(std::vector<float> output,
                                               int original_width,
                                               int original_height,
                                               int num_classes = 80,
                                               float conf_threshold = 0.25f,
                                               float nms_threshold = 0.7f,
                                               bool enable_nms = false) {
    // const int elements_per_box = 4 + num_classes;
    // std::vector<Result> detections;
    // for (size_t i = 0; i < output.size(); i += elements_per_box) {
    // }

    constexpr int elements_per_box = 6;
    std::vector<Result> detections;
    for (size_t i = 0; i < output.size(); i += elements_per_box) {
        float x = output[i];
        float y = output[i + 1];
        float w = output[i + 2];
        float h = output[i + 3];

        if (x == 0 && y == 0 && w == 0 && h == 0) {
            continue;
        }

        detections.push_back(Result{
            .class_id = static_cast<int>(output[i + 5]),
            .score = output[i + 4],
            .normalized_box = cv::Rect2f(x, y, w, h),
            .mapped_box = cv::Rect(static_cast<int>(x / input_width * original_width),
                                   static_cast<int>(y / input_height * original_height),
                                   static_cast<int>(w / input_width * original_height),
                                   static_cast<int>(h / input_height * original_height)),
        });
    }

    return detections;
}

/**
 * @brief 绘制检测结果
 * @param[in, out] img 需要绘制到的图像
 * @param[in] results
 */
inline void plot(cv::Mat& img, std::vector<Result> results) {
    AHRI_LOGGER_DEBUG("Result size: {}", results.size());
    for (auto&& result : results) {
        cv::Scalar color = Asuka::OpenCV::random_color();
        cv::rectangle(img, result.mapped_box, color, 2);

        std::string label = fmt::format("{}: {:.2f}", coco80[result.class_id], result.score);
        int baseline;
        cv::Size labelsize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(img, cv::Point(result.mapped_box.x, result.mapped_box.y - labelsize.height - baseline),
                      cv::Point(result.mapped_box.x + labelsize.width, result.mapped_box.y), color, cv::FILLED);
        cv::putText(img, label, cv::Point(result.mapped_box.x, result.mapped_box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Classes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class YOLO {
public:
    YOLO(YOLOVersion yoloversion, std::filesystem::path model_path)
        : _yoloversion(yoloversion), _model_path(model_path) {}

    ~YOLO() {}

private:
    YOLOVersion _yoloversion;
    std::filesystem::path _model_path;
};
}  // namespace Asuka::YOLO

#endif  // !ASUKA_YOLO_HPP
