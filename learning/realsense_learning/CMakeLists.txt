add_executable(realsense_base base.cpp)
target_link_libraries(realsense_base PRIVATE realsense2::fw realsense2::rsutils realsense2::realsense2 realsense2::realsense-file)

add_executable(realsense_capture capture.cpp)
target_link_libraries(realsense_capture PRIVATE realsense2::fw realsense2::rsutils realsense2::realsense2 realsense2::realsense-file)
target_link_libraries(realsense_capture PRIVATE opencv_core opencv_imgproc opencv_highgui)
