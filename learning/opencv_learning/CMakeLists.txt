add_executable(image_base image_base.cpp)
target_link_libraries(image_base PRIVATE opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(image_base PRIVATE fmt::fmt-header-only)

add_executable(camera camera.cpp)
target_link_libraries(camera PRIVATE opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(camera PRIVATE fmt::fmt-header-only)

add_executable(keypoints keypoints.cpp)
target_link_libraries(keypoints PRIVATE opencv_core opencv_imgproc opencv_highgui opencv_features2d)
target_link_libraries(keypoints PRIVATE fmt::fmt-header-only)

add_executable(line_detection line_detection.cpp)
target_link_libraries(line_detection PRIVATE opencv_core opencv_imgproc opencv_highgui opencv_ximgproc)
target_link_libraries(line_detection PRIVATE fmt::fmt-header-only)

add_executable(morphological morphological.cpp)
target_link_libraries(morphological PRIVATE opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(morphological PRIVATE fmt::fmt-header-only)

add_executable(moving_detection moving_detection.cpp)
target_link_libraries(moving_detection PRIVATE opencv_core opencv_imgproc opencv_highgui opencv_video)
target_link_libraries(moving_detection PRIVATE fmt::fmt-header-only)

add_executable(transform transform.cpp)
target_link_libraries(transform PRIVATE opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(transform PRIVATE fmt::fmt-header-only)

add_executable(opencv_main main.cpp)
target_link_libraries(opencv_main PRIVATE fmt::fmt-header-only)
target_link_libraries(opencv_main PRIVATE spdlog::spdlog_header_only)
target_link_libraries(opencv_main PRIVATE opencv_core opencv_imgproc opencv_highgui opencv_dnn)
install_exe(opencv_main)
