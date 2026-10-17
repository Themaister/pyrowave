#!/bin/bash

# Only checks out what is necessary to build standalone.
#
GRANITE_COMMIT=53090429539bf3404250d18fe58db971f0907c04

if [ -d Granite ]; then
	cd Granite
	git fetch origin
	git checkout $GRANITE_COMMIT
else
	git clone https://github.com/Themaister/Granite
	cd Granite
	git checkout $GRANITE_COMMIT
fi

cd ..

update() {
	git submodule sync $1
	git submodule update --init $1
}

cd Granite
update third_party/volk
update third_party/khronos/vulkan-headers
