#!/bin/bash

# This script should be run inside distrobox.
# distrobox create --image registry.gitlab.steamos.cloud/steamrt/steamrt4/sdk/arm64-on-amd64 --name aarch64
# distrobox enter aarch64

export PKG_CONFIG_PATH=$(pwd)/output-aarch64/lib/aarch64-linux-gnu/pkgconfig

cmake -S . -DCMAKE_BUILD_TYPE=Release \
	-Bbuild-aarch64 \
	-DPYROWAVE_DEVEL=ON \
	-DCMAKE_INSTALL_PREFIX=output-aarch64 \
	-DGRANITE_SHIPPING=ON \
	-DGRANITE_SYSTEM_SDL=OFF \
	--toolchain=/usr/share/steamrt/cmake/aarch64-linux-gnu-gcc.cmake \
	-G Ninja

ninja -C build-aarch64 install/strip
