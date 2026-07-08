#!/bin/bash

# 安装 bear
apt update && apt install -y bear

# 安装 fmt
cd /tmp
git clone --depth 1 --branch 12.1.0 https://github.com/fmtlib/fmt.git
cd fmt

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
make install

cd /tmp
rm -rf fmt/

pkg-config --modversion fmt

# 安装 spdlog
cd /tmp
git clone --depth 1 --branch v1.17.0 https://github.com/gabime/spdlog.git
cd spdlog

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DSPDLOG_FMT_EXTERNAL=ON ..
make -j$(nproc)
make install

cd /tmp
rm -rf spdlog/

pkg-config --modversion spdlog
