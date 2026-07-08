PROJECT := deepstream_learning
VERSIONL = 1.0.0
EXECUTABLE := deepstream_main

NVDS_VERSION := 8.0
DEEPSTREAM_DIR := /opt/nvidia/deepstream/deepstream
DEEPSTREAM_LIBS_DIR := $(DEEPSTREAM_DIR)/lib
DEEPSTREAM_INCS_DIR := $(DEEPSTREAM_DIR)/sources/includes
PKGS := gstreamer-1.0 gstreamer-video-1.0 gstreamer-rtsp-server-1.0 json-glib-1.0 x11

# compiler
CC := gcc
CXX := g++
NVCC := nvcc

STDC := c11
STDCXX := c++20
STDNVCC := c++20

FLAGS := -I./ -I$(DEEPSTREAM_INCS_DIR)
FLAGS += -I/usr/local/cuda/include
FLAGS += $(shell pkg-config --cflags $(PKGS))
FLAGS += -DSPDLOG_FMT_EXTERNAL
CFLAGS := -std=$(STDC) -Wall $(FLAGS)
CXXFLAGS := -std=$(STDCXX) -Wall $(FLAGS)
CUFLAGS := -std=$(STDNVCC) -Wall $(FLAGS) -Xcompiler -fPIC

# source
INCS := $(wildcard *.h) $(wildcard *.hpp) $(wildcard *.cuh)
SRCS := $(wildcard *.c) $(wildcard *.cpp) $(wildcard *.cu)
OBJS := $(SRCS:.c=.o)
OBJS := $(OBJS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# link flags
LIBS := -L/usr/local/cuda/lib64 -lcudart -lcuda
LIBS += -L$(DEEPSTREAM_LIBS_DIR) \
	-lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_smartrecord -lnvds_utils \
	-lnvds_msgbroker \
	-lyaml-cpp -lfmt -lspdlog -lm -ldl
LIBS += $(shell pkg-config --libs $(PKGS))
LIBS += -Wl,-rpath,$(DEEPSTREAM_LIBS_DIR)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CXXFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(CUFLAGS) $<

$(EXECUTABLE): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

all: $(EXECUTABLE)

format:
	mbake format Makefile

install:
	cp -v $(EXECUTABLE) $(DEEPSTREAM_LIBS_DIR)/bin

clean:
	rm -rf $(OBJS) $(EXECUTABLE)

.PHONY: all clean install format
