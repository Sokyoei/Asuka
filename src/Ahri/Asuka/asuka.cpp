#include "config.h"

#ifdef ASUKA_USE_OPENCV
#include "Ahri/Asuka/opencv_color.hpp"
#include "Ahri/Asuka/opencv_utils.hpp"
#endif

#ifdef ASUKA_USE_TENSORRT
#include "Ahri/Asuka/tensorrt_macro.hpp"
#include "Ahri/Asuka/tensorrt_utils.hpp"
#endif

#ifdef ASUKA_USE_ONNXRUNTIME
#include "Ahri/Asuka/onnx_utils.hpp"
#endif

#ifdef ASUKA_USE_OPENVINO
#include "Ahri/Asuka/openvino_utils.hpp"
#endif

#ifdef ASUKA_USE_REALSENSE2
#include "Ahri/Asuka/realsense2_utils.hpp"
#endif
