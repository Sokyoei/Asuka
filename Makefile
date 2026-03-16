PROJECT=Ahri.Asuka
VERSION=0.0.1

CC=gcc
CXX=g++
STDC=c11
STDCXX=c++20

CFLAGS=-std=$(STDC) -Wall
CXXFLAGS=-std=$(STDCXX) -Wall

ROOT=$(shell pwd)

export CC CXX CFLAGS CXXFLAGS ROOT

all: opencv_learn onnx_learn opencv_learn-% onnx_learn-%

# 或者 $(MAKE) -C opencv_learn
opencv_learn:
	cd opencv_learn && $(MAKE)

opencv_learn-%:
	$(MAKE) -C opencv_learn $*

onnx_learn:
	cd onnx_learn && $(MAKE)

onnx_learn-%:
	$(MAKE) -C onnx_learn $*

clean: opencv_learn onnx_learn
	cd opencv_learn && $(MAKE) clean
	cd onnx_learn && $(MAKE) clean

.PHONY: all clean opencv_learn onnx_learn opencv_learn-% onnx_learn-%
