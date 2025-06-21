// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include "pyrowave_config.hpp"

namespace Vulkan
{
class Device;
class ImageView;
class CommandBuffer;
}

namespace PyroWave
{
class Decoder
{
public:
	Decoder();
	~Decoder();

	bool init(Vulkan::Device *device, int width, int height, ChromaSubsampling chroma);
	void clear();
	bool push_packet(const void *data, size_t size);
	bool decode(Vulkan::CommandBuffer &cmd, const ViewBuffers &views);
	bool decode_is_ready(bool allow_partial_frame) const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}