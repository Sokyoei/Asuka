#include <algorithm>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

namespace Ahri {
enum class BackgroundMode { GMM, KNN };

class MovingDetector {
public:
    MovingDetector(BackgroundMode mode = BackgroundMode::GMM) {
        if (mode == BackgroundMode::GMM) {
            _model = cv::createBackgroundSubtractorMOG2();
        } else if (mode == BackgroundMode::KNN) {
            _model = cv::createBackgroundSubtractorKNN();
        } else {
            throw std::runtime_error("unsupported BackgroundMode " + std::to_string(int(mode)) + ".");
        }
        _kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        // self.landing_point = []
        // self.is_landing_point = False
        // self.current_y = 0
        // self.x_list = []
        // self.y_list = []
        // self.ignore_center_points = []
        // self.is_get_ignore_center_points = False
    }

    ~MovingDetector() { _model.release(); }

    void process(std::string video_path) {
        auto cap = cv::VideoCapture(video_path);
        cv::namedWindow("frame", cv::WINDOW_FREERATIO);

        while (true) {
            bool ret = cap.read(_frame);
            if (!ret) {
                break;
            }

            //  if not self.is_get_ignore_center_points:
            //      self.get_ignore_center_points(frame.shape[:-1])
            //      self.is_get_ignore_center_points = True
            _model->apply(_frame, _mask);
            cv::morphologyEx(_mask, _mask, cv::MORPH_OPEN, _kernel, cv::Point(-1, -1), 2);
            cv::findContours(_mask, _contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (!_contours.empty()) {
                // contour
                std::vector<double> areas;
                for (auto&& contour : _contours) {
                    areas.emplace_back(cv::contourArea(contour));
                }
                auto max_value = std::max_element(areas.cbegin(), areas.cend());
                auto max_value_index = std::distance(areas.cbegin(), max_value);
                auto max_contour = _contours.at(max_value_index);
                cv::drawContours(_frame, _contours, max_value_index, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);

                // centroid
                auto M = cv::moments(max_contour);
                if (M.m00 != 0) {
                    int cx = int(M.m10 / M.m00);
                    int cy = int(M.m01 / M.m00);
                    cv::circle(_frame, cv::Point(cx, cy), 2, cv::Scalar(0, 255, 0), 2);
                }

                // if [cx, cy] not in self.ignore_center_points:
                //      if cy > self.current_y and not self.is_landing_point:
                //          self.landing_point = [cx, cy]
                //      if cy < self.current_y:
                //          self.is_landing_point = True
                //          self.x_list.append(n)
                //          self.y_list.append(cy)
                //          self.current_y = cy
            }
            // cv2.putText(frame, f"{self.landing_point}", [0, 25], cv2.FONT_HERSHEY_COMPLEX, 1, [0, 255, 0], 2)
            cv::imshow("frame", _frame);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }

    void get_ignore_center_points() {
        constexpr int XY_LIMITS = 3;
        // height, width = frame_size
        // center_y, center_x = int(height / 2), int(width / 2)
        // center_x_list = [i for i in range(center_x - XY_LIMITS, center_x + XY_LIMITS + 1)]
        // center_y_list = [i for i in range(center_y - XY_LIMITS, center_y + XY_LIMITS + 1)]
        // self.ignore_center_points = [list(i) for i in itertools.product(center_x_list, center_y_list)]
    }

private:
    cv::Ptr<cv::BackgroundSubtractor> _model;
    cv::Mat _kernel;
    cv::Mat _frame;
    cv::Mat _mask;
    std::vector<std::vector<cv::Point>> _contours;
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto moving = Ahri::MovingDetector();
    moving.process(std::string(R"(D:\Download\throwing.mp4)"));

    return 0;
}
