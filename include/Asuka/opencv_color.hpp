/**
 * @file color.hpp
 * @date 2024/07/15
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef ASUKA_OPENCV_COLOR_HPP
#define ASUKA_OPENCV_COLOR_HPP

#include <random>

#include <opencv2/opencv.hpp>

namespace Asuka::OpenCV {
#define RED cv::Scalar(0, 0, 255)
#define GREEN cv::Scalar(0, 255, 0)
#define BLUE cv::Scalar(255, 0, 0)
#define BLACK cv::Scalar(0, 0, 0)
#define WHITE cv::Scalar(255, 255, 255)
#define YELLOW cv::Scalar(0, 255, 255)
#define VIOLET cv::Scalar(238, 130, 238)
#define PINK cv::Scalar(203, 192, 255)
#define DEEPPINK cv::Scalar(147, 20, 255)
#define PURPLE cv::Scalar(128, 0, 128)
#define SKYBLUE cv::Scalar(230, 216, 173)
#define GOLD cv::Scalar(10, 215, 255)
#define DARKGRAY cv::Scalar(169, 169, 169)

/**
 * @brief 生成随机的OpenCV Scalar颜色
 * @return cv::Scalar 随机颜色值(RGB)
 * @note 在 header-only 项目中确保随机数生成器只初始化一次(在单个编译单元内，跨 DLL 调用不行)
 */
inline cv::Scalar random_color() {
    struct RandomGenerator {
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_int_distribution<int> dis;

        RandomGenerator() : gen(rd()), dis(0, 255) {}
    };

    // 局部静态变量，在第一次调用时初始化，且仅初始化一次
    static RandomGenerator rng;

    return cv::Scalar(rng.dis(rng.gen), rng.dis(rng.gen), rng.dis(rng.gen));
}

}  // namespace Asuka::OpenCV

#endif  // !ASUKA_OPENCV_COLOR_HPP
