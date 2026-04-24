add_executable(ffmpeg_main main.cpp)
target_include_directories(ffmpeg_main PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(ffmpeg_main PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(ffmpeg_main PRIVATE ${FFMPEG_LIBRARIES})
