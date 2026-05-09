add_executable(pcl_main main.cpp)
target_link_libraries(pcl_main PRIVATE ${PCL_LIBRARIES})
target_link_libraries(pcl_main PRIVATE fmt::fmt-header-only)
