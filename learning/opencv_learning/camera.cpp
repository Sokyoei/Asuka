#include <iostream>

#include <opencv2/opencv.hpp>

namespace Ahri {
void open_camera() {
    auto video = cv::VideoCapture(0);
    auto fps = video.get(cv::CAP_PROP_FPS);

    while (true) {
        cv::Mat frame;
        bool ret = video.read(frame);
        if (!ret) {
            break;
        }

        cv::putText(frame, std::to_string(fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),
                    1, cv::LINE_AA);
        cv::imshow("camera", frame);
        auto c = cv::waitKey(1);
        if (c == 27) {
            break;
        }
    }
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::open_camera();

    return 0;
}
