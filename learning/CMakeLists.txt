if(FFmpeg AND FFMPEG_FOUND)
    add_subdirectory(ffmpeg_learning)
endif(FFmpeg AND FFMPEG_FOUND)

if(GStreamer AND GSTREAMER_FOUND)
    add_subdirectory(gstreamer_learning)
endif(GStreamer AND GSTREAMER_FOUND)
