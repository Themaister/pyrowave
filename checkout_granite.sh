#!/bin/bash

# Only checks out what is necessary to build standalone.
#
GRANITE_COMMIT=44362775d36e0c4139352f83efd96bab4e239f66

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
