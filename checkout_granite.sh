#!/bin/bash

# Only checks out what is necessary to build standalone.
#
GRANITE_COMMIT=51757b73d85bc8b632eb00d7f52645f534eef580

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
