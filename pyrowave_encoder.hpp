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
class Buffer;
class ImageView;
class CommandBuffer;
}

namespace PyroWave
{
class Encoder
{
public:
	Encoder();
	~Encoder();

	struct BitstreamBuffers
	{
		struct
		{
			const Vulkan::Buffer *buffer;
			uint64_t offset;
			uint64_t size;
		} meta, bitstream;
		size_t target_size;
	};

	bool init(Vulkan::Device *device, int width, int height, ChromaSubsampling chroma);
	bool encode(Vulkan::CommandBuffer &cmd, const ViewBuffers &views, const BitstreamBuffers &buffers);

	// Debug hackery
	const Vulkan::ImageView &get_wavelet_band(int component, int level);
	bool encode_pre_transformed(Vulkan::CommandBuffer &cmd, const BitstreamBuffers &buffers, float quant_scale);
	//

	uint64_t get_meta_required_size() const;

	struct Packet
	{
		size_t offset;
		size_t size;
	};

	size_t compute_num_packets(const void *mapped_meta, size_t packet_boundary) const;
	size_t packetize(Packet *packets, size_t packet_boundary,
					 void *bitstream, size_t size,
					 const void *mapped_meta, const void *mapped_bitstream) const;

	void report_stats(const void *mapped_meta, const void *mapped_bitstream) const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
}