#include <filesystem>

#include <opencv2/opencv.hpp>

#include "Asuka.hpp"

int main(int argc, char const* argv[]) {
    // auto img_path = (std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Popstar Ahri.jpg").string();
    /// @warning CJK 路径，不建议这么做，尽量使用 ASCII 路径
    auto img_path = (std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/星之守护者 永绽盛芒 阿狸.jpg").string();
    cv::namedWindow("Ahri", cv::WINDOW_FREERATIO);
    cv::Mat src = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::imshow("Ahri", src);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
