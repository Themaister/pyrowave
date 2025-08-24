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

	// Fragment path is optimized for typical mobile GPUs which have weak compute support.
	// iDWT is instead computed entirely in traditional render passes and fragment shaders.
	// This path is *not* recommended for desktop-class chips.
	bool init(Vulkan::Device *device, int width, int height,
	          ChromaSubsampling chroma, bool fragment_path = false);

	void clear();
	bool push_packet(const void *data, size_t size);

	// If fragment path is enabled, the command buffer must support graphics operations.
	// To synchronize, synchronize with COLOR_OUTPUT / COLOR_ATTACHMENT_WRITE / COLOR_ATTACHMENT_OPTIMAL.
	// Views must be created with VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT.
	bool decode(Vulkan::CommandBuffer &cmd, const ViewBuffers &views);

	bool decode_is_ready(bool allow_partial_frame) const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}