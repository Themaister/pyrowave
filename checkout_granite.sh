#!/bin/bash

# Only checks out what is necessary to build standalone.
#
GRANITE_COMMIT=d0bccdf9b8dd5c9969a0201b0da67bd7e86cfee2

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
