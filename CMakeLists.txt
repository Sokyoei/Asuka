cmake_minimum_required(VERSION 3.20.0)

project(Asuka VERSION 0.0.1)

########################################################################################################################
# languages
########################################################################################################################
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
endif(CMAKE_CUDA_COMPILER)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

########################################################################################################################
# options
########################################################################################################################
option(ONNXRuntime          "enable ONNXRuntime"                OFF)
option(TensorRT             "enable TensorRT"                   OFF)
option(OpenVINO             "enable Intel OpenVINO"             OFF)
option(IntelRealSense       "enable Intel IntelRealSense"       OFF)
option(AsukaModule          "build Asuka for C++20 module"      OFF)

########################################################################################################################
# CMAKE_PREFIX_PATH/CMAKE_MODULE_PATH set
########################################################################################################################
# set ONNXRuntime cmake dir
if(ONNXRuntime)
    set(CMAKE_PREFIX_PATH ${ONNXRuntime_CMAKE_DIR} ${CMAKE_PREFIX_PATH})
endif(ONNXRuntime)

# set OpenVINO cmake dir
if(OpenVINO)
    set(CMAKE_PREFIX_PATH ${OpenVINO_CMAKE_DIR} ${CMAKE_PREFIX_PATH})
endif(OpenVINO)

message(STATUS "cmake prefix path: ${CMAKE_PREFIX_PATH}")
# project FindXXX.cmake PATH
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

########################################################################################################################
include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/library.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

set(ASUKA_ROOT ${PROJECT_SOURCE_DIR})

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    # clone https://github.com/Sokyoei/Ceceilia in Vison same level directory
    ${PROJECT_SOURCE_DIR}/../Ceceilia
    ${PROJECT_SOURCE_DIR}/../Ceceilia/include
)

########################################################################################################################
# Ahri::Asuka and Ahri::Asuka-Header-Only
########################################################################################################################
file(
    GLOB_RECURSE
    ASUKA_SOURCE_FILES
    "src/*.cpp"
    "include/*.hpp"
)
add_library(ahri_asuka SHARED ${ASUKA_SOURCE_FILES})
target_link_libraries(ahri_asuka PUBLIC $<$<BOOL:${USE_FMT}>:fmt::fmt>)
target_link_libraries(ahri_asuka PUBLIC $<$<BOOL:${USE_SPDLOG}>:spdlog::spdlog>)
target_link_libraries(ahri_asuka PUBLIC $<$<BOOL:${USE_TENSORRT}>:TensorRT::TensorRT>)
target_include_directories(ahri_asuka PUBLIC $<$<BOOL:${USE_TENSORRT}>:${CUDAToolkit_INCLUDE_DIRS}>)
target_link_libraries(ahri_asuka PUBLIC $<$<BOOL:${USE_ONNXRUNTIME}>:ONNXRuntime::ONNXRuntime>)
target_link_libraries(ahri_asuka PUBLIC $<$<BOOL:${USE_OPENVINO}>:openvino::runtime;openvino::frontend::onnx;openvino::threading>)
target_link_libraries(ahri_asuka PUBLIC $<$<BOOL:${USE_OPENCV}>:opencv_core;opencv_imgproc;opencv_highgui;opencv_dnn>)
target_link_options(ahri_asuka PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/OPT:NOREF>)
target_compile_definitions(ahri_asuka PUBLIC AHRI_ASUKA_BUILD_SHARED_LIBRARY)
add_library(Ahri::Asuka ALIAS ahri_asuka)

add_library(ahri_asuka_header_only INTERFACE)
target_include_directories(ahri_asuka_header_only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
add_library(Ahri::Asuka-Header-Only ALIAS ahri_asuka_header_only)

########################################################################################################################
# Learning projects
########################################################################################################################
if(OpenCV_FOUND)
    add_subdirectory(opencv_learn)
endif(OpenCV_FOUND)

if(ONNXRuntime AND ONNXRuntime_FOUND)
    add_subdirectory(onnx_learn)
endif(ONNXRuntime AND ONNXRuntime_FOUND)

if(TensorRT AND TensorRT_FOUND)
    add_subdirectory(tensorrt_learn)
endif(TensorRT AND TensorRT_FOUND)

if(IntelRealSense AND realsense2_FOUND)
    add_subdirectory(realsense_learn)
endif(IntelRealSense AND realsense2_FOUND)

if(OpenVINO AND OpenVINO_FOUND)
    add_subdirectory(openvino_learn)
endif(OpenVINO AND OpenVINO_FOUND)

# config.h settings
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

include(${PROJECT_SOURCE_DIR}/cmake/install.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/package.cmake)
