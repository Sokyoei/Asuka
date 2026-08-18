#ifndef CONFIG_H
#define CONFIG_H

#cmakedefine ASUKA_ROOT "@ASUKA_ROOT@"

// CUDA support
#cmakedefine ASUKA_HAVE_CUDA

// third libraries
#cmakedefine ASUKA_USE_FMT
#cmakedefine ASUKA_USE_SPDLOG
#cmakedefine ASUKA_USE_OPENCV
#cmakedefine ASUKA_USE_ONNXRUNTIME
#cmakedefine ASUKA_USE_TENSORRT
#cmakedefine ASUKA_USE_OPENVINO
#cmakedefine ASUKA_USE_REALSENSE2
#cmakedefine ASUKA_USE_FFMPEG
#cmakedefine ASUKA_USE_GLAD
#cmakedefine ASUKA_USE_GLFW3
#cmakedefine ASUKA_USE_IMGUI
#cmakedefine ASUKA_USE_GSTREAMER
#cmakedefine ASUKA_USE_PCL
#cmakedefine ASUKA_USE_CUDATOOLKIT
#cmakedefine ASUKA_USE_EIGEN3

#endif  // !CONFIG_H
