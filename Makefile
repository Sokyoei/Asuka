PROJECT := Asuka
VERSION = 1.0.0

ROOT := $(shell pwd)
# https://github.com/Sokyoei/Ceceilia
CECEILIA_DIR ?= ../Ceceilia
CECEILIA_DIR := $(abspath $(CECEILIA_DIR))
PKGS := fmt spdlog

# compiler and flags
CC := gcc
CXX := g++
NVCC := nvcc
STDC := c11
STDCXX := c++20
STDNVCC := c++20

PKG_CFLAGS := $(shell pkg-config --cflags $(PKGS))
PKG_LIBS := $(shell pkg-config --libs $(PKGS))

CPPFLAGS := -I$(ROOT) -I$(ROOT)/include/ $(PKG_CFLAGS) -I$(CECEILIA_DIR)/include/
CPPFLAGS += -DSPDLOG_FMT_EXTERNAL
CFLAGS := -std=$(STDC) -Wall
CXXFLAGS := -std=$(STDCXX) -Wall
CUFLAGS := -std=$(STDNVCC) -Wall -Xcompiler -fPIC
LDFLAGS :=
LIBS := $(PKG_LIBS)

export CC CXX NVCC CPPFLAGS CFLAGS CXXFLAGS CUFLAGS LDFLAGS LIBS ROOT

EXCLUDE_DIRS := deepstream_learning
VALID_SUBDIRS := $(shell find learning -maxdepth 1 -type d -name '*_learning' -exec test -f {}/Makefile \; -print)
VALID_SUBDIRS := $(filter-out $(addprefix learning/, $(EXCLUDE_DIRS)), $(VALID_SUBDIRS))
VALID_TARGETS := $(patsubst %_learning, %_main, $(notdir $(VALID_SUBDIRS)))

all: $(VALID_TARGETS)

%_main:
	@dir=learning/$*_learning; \
	if [ -d "$$dir" ] && [ -f "$$dir/Makefile" ]; then \
		$(MAKE) -C $$dir $@; \
	else \
		echo "Warning: $$dir/Makefile not found, skipping target $@"; \
	fi

format:
	mbake format Makefile

	@for dir in $(VALID_SUBDIRS); do \
		if [ -f "$$dir/Makefile" ]; then \
			echo "Formatting $$dir/Makefile"; \
			(cd $$dir && mbake format Makefile); \
		fi \
	done

clean:
	@for dir in $(VALID_SUBDIRS); do \
		if [ -d "$$dir" ] && [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C $$dir clean; \
		fi \
	done

.PHONY: all clean %_main format
