if(FFmpeg AND FFMPEG_FOUND)
    add_subdirectory(ffmpeg_learning)
endif(FFmpeg AND FFMPEG_FOUND)

if(GStreamer AND GSTREAMER_FOUND)
    add_subdirectory(gstreamer_learning)
endif(GStreamer AND GSTREAMER_FOUND)

if(PCL AND PCL_FOUND)
    add_subdirectory(pcl_learning)
endif(PCL AND PCL_FOUND)

if(RealSense AND realsense2_FOUND)
    add_subdirectory(realsense_learning)
endif(RealSense AND realsense2_FOUND)
