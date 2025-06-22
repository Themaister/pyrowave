// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include <string.h>

#include "global_managers_init.hpp"
#include "application.hpp"
#include "filesystem.hpp"
#include "device.hpp"
#include "context.hpp"
#include "thread_group.hpp"
#include "pyrowave_encoder.hpp"
#include "pyrowave_decoder.hpp"
#include "pyrowave_common.hpp"
#include <random>
#include "fft.hpp"
#include "yuv4mpeg.hpp"
#include "shaders/slangmosh.hpp"
#include "math.hpp"
#include "muglm/muglm_impl.hpp"

using namespace Granite;
using namespace Vulkan;

static void run_encoder_test(Device &device,
                             PyroWave::Encoder &enc,
                             PyroWave::Decoder &dec,
                             const PyroWave::ViewBuffers &inputs,
                             const PyroWave::ViewBuffers &outputs,
							 size_t bitstream_size, YUV4MPEGFile &f)
{
	BufferCreateInfo buffer_info = {};
	buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	buffer_info.size = enc.get_meta_required_size();
	buffer_info.domain = BufferDomain::Device;
	auto meta = device.create_buffer(buffer_info);
	buffer_info.domain = BufferDomain::CachedHost;
	auto meta_host = device.create_buffer(buffer_info);

	buffer_info.size = bitstream_size + 2 * enc.get_meta_required_size();
	buffer_info.domain = BufferDomain::Device;
	auto bitstream = device.create_buffer(buffer_info);
	buffer_info.domain = BufferDomain::CachedHost;
	auto bitstream_host = device.create_buffer(buffer_info);

	PyroWave::Encoder::BitstreamBuffers buffers = {};
	buffers.meta.buffer = meta.get();
	buffers.meta.size = meta->get_create_info().size;
	buffers.bitstream.buffer = bitstream.get();
	buffers.bitstream.size = bitstream->get_create_info().size;
	buffers.target_size = bitstream_size;

	{
		auto cmd = device.request_command_buffer();
#if 1
		enc.encode(*cmd, inputs, buffers);
#else
		cmd->image_barrier(enc.get_wavelet_band(0, 0).get_image(),
		                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
		                   VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

		cmd->clear_image(enc.get_wavelet_band(0, 0).get_image(), {});

		cmd->barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
		             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

		constexpr int w = 32;
		constexpr int h = 32;

		auto *coeffs = static_cast<uint16_t *>(
				cmd->update_image(enc.get_wavelet_band(0, 0).get_image(),
				                  {}, { w, h, 1 }, w, 256, { VK_IMAGE_ASPECT_COLOR_BIT, 3, 1, 1 }));

		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
				coeffs[w * y + x] = floatToHalf((float(x) + (x ? 0.5f : 0.0f)) * (y & 1 ? -1.0f : 1.0f));

		cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
		             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

		enc.encode_pre_transformed(*cmd, buffers, 1.0f);
#endif

		cmd->copy_buffer(*bitstream_host, *bitstream);
		cmd->copy_buffer(*meta_host, *meta);
		cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
		             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

		Fence fence;
		device.submit(cmd, &fence);
		device.next_frame_context();
		fence->wait();
	}

	auto *mapped_meta = static_cast<const PyroWave::BitstreamPacket *>(
			device.map_host_buffer(*meta_host, MEMORY_ACCESS_READ_BIT));
	auto *mapped_bits = static_cast<const uint32_t *>(
			device.map_host_buffer(*bitstream_host, MEMORY_ACCESS_READ_BIT));

	std::vector<uint8_t> reordered_packet_buffer(8 * 1024 * 1024);
	size_t num_packets = enc.compute_num_packets(mapped_meta, 8 * 1024);
	std::vector<PyroWave::Encoder::Packet> packets(num_packets);
	size_t out_packets = enc.packetize(packets.data(), 8 * 1024,
	                                   reordered_packet_buffer.data(),
	                                   reordered_packet_buffer.size(),
	                                   mapped_meta, mapped_bits);
	enc.report_stats(mapped_meta, mapped_bits);
	(void)out_packets;

	size_t encoded_size = 0;
	for (auto &p : packets)
		encoded_size += p.size;

	LOGI("Total encoded size: %zu\n", encoded_size);

	if (encoded_size > bitstream_size)
	{
		LOGE("Broken rate control\n");
		return;
	}

	assert(out_packets == num_packets);

#if 0
	struct DummyPacket
	{
		alignas(uint32_t) PyroWave::BitstreamHeader header;
		uint16_t code[16];
		uint8_t q[16];
		uint8_t planes[16];
		uint8_t signs[16];
	};

	DummyPacket packet = {};
	packet.header.payload_words = sizeof(DummyPacket) / sizeof(uint32_t);
	packet.header.ballot = 0xffff;
	packet.header.quant_code = PyroWave::encode_quant(1.0f);
	for (auto &c : packet.code)
		c = 0x1;
	for (auto &q : packet.q)
		q = 6 << 4;
	for (auto &p : packet.planes)
		p = 0x7;
	packet.signs[0] = 0xff;
	packet.signs[1] = 0xff;
	packet.signs[2] = 0xff;
	packet.signs[3] = 0xff;
	packet.signs[4] = 0xff;
	dec.push_packet(&packet, sizeof(packet));
#endif

#if 1
	for (auto &p : packets)
		if (!dec.push_packet(reordered_packet_buffer.data() + p.offset, p.size))
			return;
#endif

	BufferHandle out_buffers[3];
	int bytes_per_pixel = YUV4MPEGFile::format_to_bytes_per_component(f.get_format());
	for (int i = 0; i < 3; i++)
	{
		BufferCreateInfo info = {};
		info.size = outputs.planes[i]->get_view_width() * outputs.planes[i]->get_view_height() * bytes_per_pixel;
		info.domain = BufferDomain::CachedHost;
		info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		out_buffers[i] = device.create_buffer(info);
	}

	{
		auto cmd = device.request_command_buffer();
		//if (!dec.decode_is_ready(false))
		//	return;
		if (!dec.decode(*cmd, outputs))
			return;

		for (int i = 0; i < 3; i++)
		{
			cmd->image_barrier(outputs.planes[i]->get_image(),
			                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
		}

		for (int i = 0; i < 3; i++)
		{
			cmd->copy_image_to_buffer(*out_buffers[i], outputs.planes[i]->get_image(), 0, {},
			                          { outputs.planes[i]->get_view_width(), outputs.planes[i]->get_view_height(), 1 },
			                          0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
		}

		cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
					 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

		Fence fence;
		device.submit(cmd, &fence);
		device.next_frame_context();
		fence->wait();

		if (!f.begin_frame())
			return;

		for (auto &buf : out_buffers)
		{
			const void *mapped = device.map_host_buffer(*buf, MEMORY_ACCESS_READ_BIT);
			if (!f.write(mapped, buf->get_create_info().size))
			{
				LOGE("Failed to write plane.\n");
				return;
			}
		}
	}
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

struct BlockCounts
{
	int offset;
	int count;
};

static void run_vulkan_test(Device &device, const char *in_path, const char *out_path, size_t bitstream_size)
{
	YUV4MPEGFile input, output;

	if (!input.open_read(in_path))
		return;

	if (!output.open_write(out_path, input.get_params()))
		return;

	auto width = input.get_width();
	auto height = input.get_height();

	auto fmt = YUV4MPEGFile::format_to_bytes_per_component(input.get_format()) == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
	auto chroma = YUV4MPEGFile::format_has_subsampling(input.get_format()) ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444;
	auto inputs = create_ycbcr_images(device, width, height, fmt, chroma);
	auto outputs = create_ycbcr_images(device, width, height, fmt, chroma);

	PyroWave::Encoder enc;
	if (!enc.init(&device, width, height, chroma))
		return;

	PyroWave::Decoder dec;
	if (!dec.init(&device, width, height, chroma))
		return;

	bool has_rdoc = Device::init_renderdoc_capture();
	unsigned frames = 0;

	if (has_rdoc)
		device.begin_renderdoc_capture();

	for (;;)
	{
		if (!input.begin_frame())
			break;

		auto cmd = device.request_command_buffer();

		for (int i = 0; i < 3; i++)
		{
			cmd->image_barrier(*inputs.images[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
							   0, 0,
							   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
			cmd->image_barrier(*outputs.images[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                   0, 0,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}

		for (int i = 0; i < 3; i++)
		{
			auto *y = cmd->update_image(*inputs.images[i]);
			if (!input.read(y, inputs.images[i]->get_width() * inputs.images[i]->get_height() * YUV4MPEGFile::format_to_bytes_per_component(input.get_format())))
			{
				LOGE("Failed to read plane.\n");
				device.submit_discard(cmd);
				return;
			}
		}

		for (int i = 0; i < 3; i++)
		{
			cmd->image_barrier(*inputs.images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}

		device.submit(cmd);
		run_encoder_test(device, enc, dec, inputs.views, outputs.views, bitstream_size, output);
		frames++;
		if (has_rdoc && frames >= 10)
			break;
	}

	if (has_rdoc)
		device.end_renderdoc_capture();
}

static void run_vulkan_test(const char *in_path, const char *out_path, size_t bitstream_size)
{
	Global::init(Global::MANAGER_FEATURE_EVENT_BIT | Global::MANAGER_FEATURE_FILESYSTEM_BIT |
	             Global::MANAGER_FEATURE_THREAD_GROUP_BIT, 1);

	Filesystem::setup_default_filesystem(GRANITE_FILESYSTEM(), ASSET_DIRECTORY);

	if (!Context::init_loader(nullptr))
		return;

	Context::SystemHandles handles = {};
	handles.thread_group = GRANITE_THREAD_GROUP();
	handles.filesystem = GRANITE_FILESYSTEM();

	Context ctx;
	ctx.set_system_handles(handles);

	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0, CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return;

	Device dev;
	dev.set_context(ctx);

	//run_noise_power_test(dev);
	run_vulkan_test(dev, in_path, out_path, bitstream_size);
}

int main(int argc, char **argv)
{
	if (argc != 4)
		return EXIT_FAILURE;

	run_vulkan_test(argv[1], argv[2], strtoul(argv[3], nullptr, 0));
}
