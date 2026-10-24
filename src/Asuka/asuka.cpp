#include "config.h"

#ifdef ASUKA_USE_OPENCV
#include "Asuka/opencv_color.hpp"
#include "Asuka/opencv_utils.hpp"
#endif

#ifdef ASUKA_USE_TENSORRT
#include "Asuka/tensorrt_macro.hpp"
#include "Asuka/tensorrt_utils.hpp"
#endif

#ifdef ASUKA_USE_ONNXRUNTIME
#include "Asuka/onnx_utils.hpp"
#endif

#ifdef ASUKA_USE_OPENVINO
#include "Asuka/openvino_utils.hpp"
#endif

#ifdef ASUKA_USE_REALSENSE2
#include "Asuka/realsense2_utils.hpp"
#endif
