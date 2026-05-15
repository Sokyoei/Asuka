#include "Ahri/Asuka/realsense2_utils.hpp"

int main(int argc, char const* argv[]) {
    int ret = Ahri::RealSense2::check_device();
    return ret;
}
