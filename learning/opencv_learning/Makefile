CC ?= gcc
CXX ?= g++
NVCC ?= nvcc

EXECUTABLE := opencv_main

INCS := $(wildcard *.h) $(wildcard *.hpp) $(wildcard *.cuh)
SRCS := $(wildcard *.c) $(wildcard *.cpp) $(wildcard *.cu)
OBJS := $(SRCS:.c=.o)
OBJS := $(OBJS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

PKGS := opencv4
include $(ROOT)/common.mk

$(EXECUTABLE): main.o Makefile
	$(CXX) -o $@ main.o $(LIBS) $(LDFLAGS)

all: $(EXECUTABLE)

clean:
	rm -f $(OBJS) $(EXECUTABLE)

.PHONY: clean all
