#ifndef CONFIG_H
#define CONFIG_H

#cmakedefine ASUKA_ROOT "@ASUKA_ROOT@"

// CUDA support
#cmakedefine HAS_CUDA

// third libraries
#cmakedefine USE_FMT
#cmakedefine USE_SPDLOG
#cmakedefine USE_OPENCV
#cmakedefine USE_ONNXRUNTIME
#cmakedefine USE_TENSORRT
#cmakedefine USE_OPENVINO
#cmakedefine USE_REALSENSE2
#cmakedefine USE_FFMPEG
#cmakedefine USE_GLAD
#cmakedefine USE_GLFW3
#cmakedefine USE_IMGUI

#endif  // !CONFIG_H
