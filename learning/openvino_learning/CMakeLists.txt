add_library(ahri_plugin_openvino SHARED plugins/ahri_plugin_openvino.cpp)
target_link_libraries(ahri_plugin_openvino PRIVATE openvino::runtime openvino::frontend::onnx openvino::threading)

add_executable(openvino_main main.cpp)
target_link_libraries(openvino_main PRIVATE openvino::runtime openvino::frontend::onnx openvino::threading)
target_link_libraries(openvino_main PRIVATE ahri_plugin_openvino)
target_link_libraries(openvino_main PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)
target_link_libraries(openvino_main PRIVATE opencv_core opencv_imgproc opencv_highgui opencv_dnn)
