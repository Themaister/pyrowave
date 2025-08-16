#!/bin/bash

./Granite/tools/create_android_build.py \
	--output-gradle android \
	--application-id net.themaister.pyrowave_viewer \
	--granite-dir Granite \
	--native-target pyrowave-viewer \
	--app-name "PyroWave Viewer" \
	--abis arm64-v8a \
	--cmake-lists-toplevel CMakeLists.txt \
	--assets shaders \
	--builtin Granite/assets
echo org.gradle.jvmargs=-Xmx4096M >> gradle.properties
