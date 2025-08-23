// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include <string.h>

#include "global_managers_init.hpp"
#include "device.hpp"
#include "context.hpp"
#include "pyrowave_decoder.hpp"
#include "pyrowave_common.hpp"
#include "yuv4mpeg.hpp"
#include "shaders/slangmosh.hpp"

using namespace Granite;
using namespace Vulkan;

struct DecodedBuffer
{
	BufferHandle planes[3];
	Fence fence;
};

static DecodedBuffer run_decoder_frame(CommandBufferHandle &cmd,
                                       PyroWave::Decoder &dec,
                                       const PyroWave::ViewBuffers &outputs,
                                       uint32_t frame_index)
{
	auto &device = cmd->get_device();
	DecodedBuffer decoded;

	for (int i = 0; i < 3; i++)
	{
		BufferCreateInfo bufinfo = {};
		bufinfo.domain = BufferDomain::CachedHost;
		bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufinfo.size = format_get_layer_size(outputs.planes[i]->get_format(),
		                                     VK_IMAGE_ASPECT_COLOR_BIT,
		                                     outputs.planes[i]->get_view_width(),
		                                     outputs.planes[i]->get_view_height(), 1);
		decoded.planes[i] = device.create_buffer(bufinfo);
	}

	dec.decode(*cmd, outputs);

	for (auto &plane : outputs.planes)
	{
		cmd->image_barrier(plane->get_image(),
		                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
	}

	for (int i = 0; i < 3; i++)
	{
		cmd->copy_image_to_buffer(*decoded.planes[i], outputs.planes[i]->get_image(), 0,
		                          {}, { outputs.planes[i]->get_view_width(),
		                                outputs.planes[i]->get_view_height(),
		                                outputs.planes[i]->get_view_depth() },
		                          0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
	}

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	device.submit(cmd, &decoded.fence);
	device.next_frame_context();

	LOGI("Submitted frame %06u ...\n", frame_index);
	return decoded;
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

static bool write_payload(YUV4MPEGFile &file, Device &device, const DecodedBuffer &decoded)
{
	if (!file.begin_frame())
		return false;

	for (auto &plane_ptr : decoded.planes)
	{
		auto *plane = device.map_host_buffer(*plane_ptr, MEMORY_ACCESS_READ_BIT);
		if (!file.write(plane, plane_ptr->get_create_info().size))
			return false;
	}

	return true;
}

static bool read_payload(FILE *file, PyroWave::Decoder &decoder)
{
	std::vector<uint8_t> packetized_data;
	uint32_t u32_size;

	for (;;)
	{
		if (fread(&u32_size, sizeof(u32_size), 1, file) != 1)
			return false;
		packetized_data.resize(u32_size);
		if (fread(packetized_data.data(), 1, u32_size, file) != u32_size)
			return false;

		if (!decoder.push_packet(packetized_data.data(), packetized_data.size()))
			return false;

		if (decoder.decode_is_ready(false))
			return true;
	}
}

static const char *format_to_str(YUV4MPEGFile::Format fmt)
{
	switch (fmt)
	{
	case YUV4MPEGFile::Format::YUV420P:
		return "C420";
	case YUV4MPEGFile::Format::YUV420P16:
		return "C420p16";
	case YUV4MPEGFile::Format::YUV444P:
		return "C444";
	case YUV4MPEGFile::Format::YUV444P16:
		return "C444p16";
	default:
		return "???";
	}
}

static void run_decoder(Device &device, const char *out_path, const char *in_path)
{
	struct FILEDeleter { void operator()(FILE *file) { if (file) fclose(file); } };
	std::unique_ptr<FILE, FILEDeleter> infile;

	infile.reset(fopen(in_path, "rb"));
	if (!infile)
	{
		LOGE("Failed to open input file.\n");
		return;
	}

	char magic[9] = {};
	if (fread(magic, 1, 8, infile.get()) != 8)
	{
		LOGE("Failed to read magic.\n");
		return;
	}

	if (strcmp(magic, "PYROWAVE") != 0)
	{
		LOGE("Invalid magic.\n");
		return;
	}

	int32_t u32_params[8];
	if (fread(u32_params, sizeof(u32_params), 1, infile.get()) != 1)
	{
		LOGE("Failed to read parameters.\n");
		return;
	}

	PyroWave::Decoder dec;
	int width = u32_params[0];
	int height = u32_params[1];
	auto format = YUV4MPEGFile::Format(u32_params[2]);
	auto chroma = PyroWave::ChromaSubsampling(u32_params[3]);
	bool is_full = u32_params[4] != 0;
	int frame_rate_num = u32_params[5];
	int frame_rate_den = u32_params[6];
	// Unused chroma siting. YUV4MPEG doesn't seem to have proper support for that.
	if (!dec.init(&device, width, height, chroma))
		return;

	YUV4MPEGFile output;
	char params[1024];

	snprintf(params, sizeof(params), "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 XCOLORRANGE=%s %s\n",
	         width, height, frame_rate_num, frame_rate_den, is_full ? "FULL" : "LIMITED", format_to_str(format));

	if (!output.open_write(out_path, params))
	{
		LOGE("Failed to open input file.\n");
		return;
	}

	auto fmt = YUV4MPEGFile::format_to_bytes_per_component(output.get_format()) == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
	auto outputs = create_ycbcr_images(device, width, height, fmt, chroma);

	DecodedBuffer queue[2];
	uint32_t frame_index = 0;

	for (;;)
	{
		auto &q = queue[frame_index & 1];
		if (q.fence)
		{
			q.fence->wait();
			q.fence.reset();
			if (!write_payload(output, device, q))
			{
				LOGE("Failed to write payload.\n");
				break;
			}
		}

		if (!read_payload(infile.get(), dec))
			break;

		auto cmd = device.request_command_buffer();

		for (auto &img : outputs.images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, 0,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}

		queue[frame_index & 1] = run_decoder_frame(cmd, dec, outputs.views, frame_index);
		frame_index++;
	}

	frame_index--;

	auto &q = queue[frame_index & 1];
	if (q.fence)
	{
		q.fence->wait();
		if (!write_payload(output, device, q))
			LOGE("Failed to write payload.\n");
	}
}

static void run_decoder(const char *out_path, const char *in_path)
{
	if (!Context::init_loader(nullptr))
		return;

	Context ctx;

	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0, CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return;

	Device dev;
	dev.set_context(ctx);

	run_decoder(dev, out_path, in_path);
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		LOGE("Usage: pyrowave-encode <input.pyrowave> <output.y4m>\n");
		return EXIT_FAILURE;
	}

	run_decoder(argv[2], argv[1]);
}
