#!/bin/bash

# This is run from inside the container.
# Useful as an initial template.
#
mkdir -p build-steamrt
cd build-steamrt

# Can omit PYROWAVE_UTILS if just building the .so.
# Hack for now, copy over the ffmpeg build folder from pyrofling.
# Symlink won't work due to docker shenanigans.

cmake .. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=../steamrt-output \
	-DPYTHON_EXECUTABLE=$(which python3) \
	-DPYROWAVE_UTILS=ON \
	-DCMAKE_PREFIX_PATH=/pyro/ffmpeg-build-linux-steamrt/output \
	-G Ninja

ninja install/strip -v

