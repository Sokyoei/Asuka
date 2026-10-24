#include "Asuka/realsense2_utils.hpp"

int main(int argc, char const* argv[]) {
    int ret = Asuka::RealSense2::check_device();
    return ret;
}
