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
		enc.encode(*cmd, inputs, buffers);
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

	struct DummyPacket
	{
		alignas(uint32_t) PyroWave::BitstreamHeader header;
		uint16_t code[2];
		uint8_t q[2];
		uint8_t planes[4];
		uint8_t signs[1];
	};

	DummyPacket packet = {};
	packet.header.payload_words = sizeof(DummyPacket) / sizeof(uint32_t);
	packet.header.ballot = (1 << 0) | (1 << 4);
	packet.header.quant_code = PyroWave::encode_quant(1.0f);
	packet.code[0] = 0x11;
	packet.code[1] = 0x11;
	packet.q[0] = 6 << 4;
	packet.q[1] = 7 << 4;
	packet.planes[0] = 0x2;
	packet.planes[1] = 0x3;
	packet.planes[2] = 0x2;
	packet.planes[3] = 0x3;
	packet.signs[0] = 0x0e;
	dec.push_packet(&packet, sizeof(packet));

#if 0
	for (auto &p : packets)
		if (!dec.push_packet(reordered_packet_buffer.data() + p.offset, p.size))
			return;
#endif

	BufferHandle out_buffers[3];
	for (int i = 0; i < 3; i++)
	{
		BufferCreateInfo info = {};
		info.size = outputs.planes[i]->get_view_width() * outputs.planes[i]->get_view_height();
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

static YCbCrImages create_ycbcr_images(Device &device, int width, int height, VkFormat fmt = VK_FORMAT_R8_UNORM)
{
	YCbCrImages images;
	auto info = ImageCreateInfo::immutable_2d_image(width, height, fmt);
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	images.images[0] = device.create_image(info);
	device.set_name(*images.images[0], "Y");

	info.width >>= 1;
	info.height >>= 1;

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

static BlockCounts compute_block_counts(int width, int height, int sample_level, int sample_band, int sample_component = 0)
{
	width = PyroWave::align(width, PyroWave::Alignment);
	height = PyroWave::align(height, PyroWave::Alignment);

	BlockCounts counts = {};
	for (int level = 0; level < PyroWave::DecompositionLevels; level++)
	{
		for (int component = 0; component < PyroWave::NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			for (int band = (level == PyroWave::DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				int blocks_x_64x64 = ((width >> (level + 1)) + 63) / 64;
				int blocks_y_64x64 = ((height >> (level + 1)) + 63) / 64;

				counts.count = blocks_x_64x64 * blocks_y_64x64;
				if (level == sample_level && component == sample_component && band == sample_band)
					return counts;
				counts.offset += counts.count;
			}
		}
	}
	return counts;
}

static void run_noise_power_test(Device &device)
{
	Vulkan::ResourceLayout layout;
	PyroWave::Shaders<> shaders(device, layout, 0);

	constexpr int Width = 4096;
	constexpr int Height = 4096;
	auto outputs = create_ycbcr_images(device, Width, Height, VK_FORMAT_R32_SFLOAT);
	auto fft_outputs = create_ycbcr_images(device, Width / 2, Height / 2, VK_FORMAT_R32G32_SFLOAT);
	auto db_outputs = create_ycbcr_images(device, Width / (2 * 8), Height / (2 * 8), VK_FORMAT_R32_SFLOAT);

	FFT::Options opts;
	opts.mode = FFT::Mode::RealToComplex;
	opts.data_type = FFT::DataType::FP32;
	opts.Nx = 4096;
	opts.Ny = 4096;
	opts.Nz = 1;
	opts.dimensions = 2;
	opts.input_resource = FFT::ResourceType::Texture;
	opts.output_resource = FFT::ResourceType::Texture;
	FFT fft;
	if (!fft.plan(&device, opts))
		return;

	PyroWave::Decoder dec;
	if (!dec.init(&device, Width, Height))
		return;

	bool has_rdoc = Device::init_renderdoc_capture();

	if (has_rdoc)
		device.begin_renderdoc_capture();
	auto cmd = device.request_command_buffer();

	for (int i = 0; i < 3; i++)
	{
		cmd->image_barrier(*outputs.images[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                   0, 0,
		                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	}

	dec.clear();

	std::vector<uint32_t> packet_buffer;

	std::default_random_engine rnd(1234);

	const auto append_random_block = [&](int level, int band, int step_size)
	{
		// From my Linelet master thesis. Copy paste 11 years later, ah yes :D
		float horiz_midpoint = (band & 1) ? 0.75f : 0.25f;
		float vert_midpoint = (band & 2) ? 0.75f : 0.25f;

		// Normal PC monitors.
		constexpr float dpi = 96.0f;
		// Compromise between couch gaming and desktop.
		constexpr float viewing_distance = 1.5f;
		constexpr float cpd_nyquist = 0.34f * viewing_distance * dpi;

		const float cpd = std::sqrt(horiz_midpoint * horiz_midpoint + vert_midpoint * vert_midpoint) *
		                  cpd_nyquist * std::exp2(-float(level));

		const float csf = 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));

		LOGI("CSF: level %d, band %d = %10.6f, bits = %.6f\n", level, band, csf, std::log2(1.0f / csf));

		auto counts = compute_block_counts(Width, Height, level, band);
		for (int i = 0; i < counts.count; i++)
		{
			PyroWave::BitstreamHeader header = {};
			header.ballot = 0xffff;
			header.quant_code = PyroWave::encode_quant(1.0f / step_size);
			header.block_index = i + counts.offset;
			header.payload_words = 2 + 16 + 64 * 64 / (2 * sizeof(uint32_t));
			packet_buffer.resize(header.payload_words);
			memcpy(packet_buffer.data(), &header, sizeof(header));
			for (int j = 0; j < 16; j++)
				packet_buffer[2 + j] = (63u << 26) | 0xffffu;
			for (int j = 18; j < header.payload_words; j++)
				packet_buffer[j] = rnd();

			if (!dec.push_packet(packet_buffer.data(), packet_buffer.size() * sizeof(uint32_t)))
				break;
		}
	};

	append_random_block(4, 0, 4096);
	append_random_block(4, 1, 2048);
	append_random_block(4, 2, 2048);
	append_random_block(4, 3, 1024);

	append_random_block(3, 1, 1024);
	append_random_block(3, 2, 1024);
	append_random_block(3, 3, 512);

	append_random_block(2, 1, 512);
	append_random_block(2, 2, 512);
	append_random_block(2, 3, 256);

	append_random_block(1, 1, 256);
	append_random_block(1, 2, 256);
	append_random_block(1, 3, 128);

	append_random_block(0, 1, 128);
	append_random_block(0, 2, 128);
	append_random_block(0, 3, 64);

	dec.decode(*cmd, outputs.views);

	cmd->image_barrier(*outputs.images[0], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	cmd->image_barrier(*fft_outputs.images[0], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                   0, 0,
	                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	cmd->image_barrier(*db_outputs.images[0], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                   0, 0,
	                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

	FFT::Resource dst = {};
	FFT::Resource src = {};
	src.image.view = outputs.views.planes[0];
	src.image.sampler = &device.get_stock_sampler(StockSampler::NearestWrap);
	src.image.input_scale[0] = 1.0f / Width;
	src.image.input_scale[1] = 1.0f / Height;

	dst.image.view = fft_outputs.views.planes[0];
	dst.image.sampler = &device.get_stock_sampler(StockSampler::NearestWrap);
	dst.image.input_scale[0] = 1.0f / Width;
	dst.image.input_scale[1] = 1.0f / Height;

	cmd->begin_region("FFT");
	fft.execute(*cmd, dst, src);
	cmd->end_region();

	cmd->image_barrier(*fft_outputs.images[0], VK_IMAGE_LAYOUT_GENERAL,
					   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
					   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	cmd->set_program(shaders.power_to_db);
	cmd->set_texture(0, 0, fft_outputs.images[0]->get_view());
	cmd->set_storage_texture(0, 1, db_outputs.images[0]->get_view());
	cmd->dispatch(Width / (2 * 8 * 8), Height / (2 * 8 * 8), 1);

	device.submit(cmd);

	if (has_rdoc)
		device.end_renderdoc_capture();
}

static void run_vulkan_test(Device &device, const char *in_path, const char *out_path, size_t bitstream_size)
{
	YUV4MPEGFile input, output;

	if (!input.open_read(in_path))
		return;

	if (!output.open_write(out_path, input.get_params()))
		return;

	auto width = input.get_width();
	auto height = input.get_height();

	auto inputs = create_ycbcr_images(device, width, height);
	auto outputs = create_ycbcr_images(device, width, height);

	PyroWave::Encoder enc;
	if (!enc.init(&device, width, height))
		return;

	PyroWave::Decoder dec;
	if (!dec.init(&device, width, height))
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
			if (!input.read(y, inputs.images[i]->get_width() * inputs.images[i]->get_height()))
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
