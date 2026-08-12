CC ?= gcc
CXX ?= g++
NVCC ?= nvcc

EXECUTABLE := opencv_main

PKGS := opencv4
include $(ROOT)/common.mk

$(EXECUTABLE): main.o Makefile
	$(CXX) -o $@ main.o $(LIBS) $(LDFLAGS)

all: $(EXECUTABLE)

clean:
	rm -f $(OBJS) $(EXECUTABLE)

.PHONY: clean all
