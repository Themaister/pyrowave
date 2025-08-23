// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include <string.h>

#include "global_managers_init.hpp"
#include "device.hpp"
#include "context.hpp"
#include "pyrowave_encoder.hpp"
#include "yuv4mpeg.hpp"
#include "shaders/slangmosh.hpp"

using namespace Granite;
using namespace Vulkan;

struct EncodedBuffer
{
	BufferHandle payload;
	BufferHandle meta;
	Fence fence;
};

static EncodedBuffer run_encoder_frame(CommandBufferHandle &cmd,
                                       PyroWave::Encoder &enc,
                                       const PyroWave::ViewBuffers &inputs,
                                       uint32_t frame_index,
                                       uint32_t bitstream_size)
{
	auto &device = cmd->get_device();

	EncodedBuffer encoded;
	BufferCreateInfo buffer_info = {};
	buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	buffer_info.size = enc.get_meta_required_size();
	buffer_info.domain = BufferDomain::Device;
	auto meta = device.create_buffer(buffer_info);
	buffer_info.domain = BufferDomain::CachedHost;
	encoded.meta = device.create_buffer(buffer_info);

	buffer_info.size = bitstream_size + 2 * enc.get_meta_required_size();
	buffer_info.domain = BufferDomain::Device;
	auto bitstream = device.create_buffer(buffer_info);
	buffer_info.domain = BufferDomain::CachedHost;
	encoded.payload = device.create_buffer(buffer_info);

	PyroWave::Encoder::BitstreamBuffers buffers = {};
	buffers.meta.buffer = meta.get();
	buffers.meta.size = meta->get_create_info().size;
	buffers.bitstream.buffer = bitstream.get();
	buffers.bitstream.size = bitstream->get_create_info().size;
	buffers.target_size = bitstream_size;

	enc.encode(*cmd, inputs, buffers);
	cmd->copy_buffer(*encoded.payload, *bitstream);
	cmd->copy_buffer(*encoded.meta, *meta);
	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             VK_ACCESS_HOST_READ_BIT);
	device.submit(cmd, &encoded.fence);
	device.next_frame_context();

	LOGI("Submitted frame %06u ...\n", frame_index);
	return encoded;
}

struct YCbCrImages
{
	Vulkan::ImageHandle images[3];
	PyroWave::ViewBuffers views;
};

static YCbCrImages create_ycbcr_images(Device &device, int width, int height, VkFormat fmt, PyroWave::ChromaSubsampling chroma)
{
	YCbCrImages images;
	auto info = ImageCreateInfo::immutable_2d_image(width, height, fmt);
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	images.images[0] = device.create_image(info);
	device.set_name(*images.images[0], "Y");

	if (chroma == PyroWave::ChromaSubsampling::Chroma420)
	{
		info.width >>= 1;
		info.height >>= 1;
	}

	images.images[1] = device.create_image(info);
	device.set_name(*images.images[1], "Cb");

	images.images[2] = device.create_image(info);
	device.set_name(*images.images[2], "Cr");

	for (int i = 0; i < 3; i++)
		images.views.planes[i] = &images.images[i]->get_view();

	return images;
}

static bool write_payload(FILE *file, PyroWave::Encoder &encoder, Device &device, const Buffer &payload, const Buffer &meta)
{
	auto *mapped_payload = device.map_host_buffer(payload, MEMORY_ACCESS_READ_BIT);
	auto *mapped_meta = device.map_host_buffer(meta, MEMORY_ACCESS_READ_BIT);

	std::vector<uint8_t> packetized_data(payload.get_create_info().size);

	PyroWave::Encoder::Packet packet = {};
	if (encoder.packetize(&packet, payload.get_create_info().size,
	                      packetized_data.data(), packetized_data.size(),
	                      mapped_meta, mapped_payload) != 1)
	{
		LOGE("Something went terribly wrong ...\n");
		std::terminate();
	}

	uint32_t u32_size = packet.size;
	if (fwrite(&u32_size, sizeof(u32_size), 1, file) != 1)
		return false;
	return fwrite(packetized_data.data() + packet.offset, 1, packet.size, file) == packet.size;
}

static void run_encoder(Device &device, const char *out_path, const char *in_path, uint32_t bitstream_size)
{
	YUV4MPEGFile input;

	if (!input.open_read(in_path))
	{
		LOGE("Failed to open input file.\n");
		return;
	}

	struct FILEDeleter { void operator()(FILE *file) { if (file) fclose(file); } };
	std::unique_ptr<FILE, FILEDeleter> out;

	out.reset(fopen(out_path, "wb"));
	if (!out)
	{
		LOGE("Failed to open output file.\n");
		return;
	}

	if (fwrite("PYROWAVE", 1, 8, out.get()) != 8)
	{
		LOGE("Failed to write magic.\n");
		return;
	}

	int32_t width = input.get_width();
	int32_t height = input.get_height();
	auto fmt = YUV4MPEGFile::format_to_bytes_per_component(input.get_format()) == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
	auto chroma = YUV4MPEGFile::format_has_subsampling(input.get_format()) ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444;

	int32_t u32_params[8] = {
		width, height, int(input.get_format()), int(chroma), input.is_full_range(),
		input.get_frame_rate_num(), input.get_frame_rate_den(), 0 /* placeholder for unknown chroma siting */
	};

	if (fwrite(u32_params, sizeof(u32_params), 1, out.get()) != 1)
	{
		LOGE("Failed to write u32 params.\n");
		return;
	}

	auto inputs = create_ycbcr_images(device, width, height, fmt, chroma);

	PyroWave::Encoder enc;
	if (!enc.init(&device, width, height, chroma))
		return;

	EncodedBuffer queue[2];
	uint32_t frame_index = 0;

	for (;;)
	{
		auto &q = queue[frame_index & 1];
		if (q.fence)
		{
			q.fence->wait();
			q.fence.reset();
			if (!write_payload(out.get(), enc, device, *q.payload, *q.meta))
			{
				LOGE("Failed to write payload.\n");
				break;
			}
		}

		if (!input.begin_frame())
			break;

		auto cmd = device.request_command_buffer();

		for (auto &img : inputs.images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                   0, 0,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
		}

		for (auto &img : inputs.images)
		{
			auto *y = cmd->update_image(*img);
			if (!input.read(y, img->get_width() * img->get_height()))
			{
				LOGE("Failed to read plane.\n");
				device.submit_discard(cmd);
				break;
			}
		}

		for (auto &img : inputs.images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}

		queue[frame_index & 1] = run_encoder_frame(cmd, enc, inputs.views, frame_index, bitstream_size);
		frame_index++;
	}

	frame_index--;

	auto &q = queue[frame_index & 1];
	if (q.fence)
	{
		q.fence->wait();
		q.fence.reset();
		if (!write_payload(out.get(), enc, device, *q.payload, *q.meta))
			LOGE("Failed to write payload.\n");
	}
}

static void run_encoder(const char *out_path, const char *in_path, uint32_t bytes_per_frame)
{
	if (!Context::init_loader(nullptr))
		return;

	Context ctx;

	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0, CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return;

	Device dev;
	dev.set_context(ctx);

	run_encoder(dev, out_path, in_path, bytes_per_frame);
}

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		LOGE("Usage: pyrowave-encode <input.y4m> <output.pyrowave> <bytes_per_frame>\n");
		return EXIT_FAILURE;
	}

	run_encoder(argv[2], argv[1], strtoul(argv[3], nullptr, 0));
}
