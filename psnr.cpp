// Copyright (c) 2025-2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "yuv4mpeg.hpp"
#include <stdint.h>
#include <cmath>
#include "context.hpp"
#include "device.hpp"
#include "os_filesystem.hpp"
#include "global_managers_init.hpp"
#include "math.hpp"
#include "muglm/muglm_impl.hpp"
#include "pyrowave_encoder.hpp"
#include "pyrowave_decoder.hpp"

using namespace Vulkan;
using namespace Granite;
using namespace PyroWave;

static float contrast_sensitivity_function(float cpd)
{
	return 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));
}

static void roundtrip_pyrowave(
		Device &device, Encoder &encoder, Decoder &decoder,
		const ImageView &out_y, const ImageView &out_cb, const ImageView &out_cr,
		const ImageView &y, const ImageView &cb, const ImageView &cr,
		size_t target_size)
{
	target_size &= ~size_t(3);

	ViewBuffers views = {};
	views.planes[0] = &y;
	views.planes[1] = &cb;
	views.planes[2] = &cr;

	BufferCreateInfo bufinfo = {};
	bufinfo.size = target_size + encoder.get_meta_required_size();
	bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufinfo.domain = BufferDomain::Device;
	auto bitstream_gpu = device.create_buffer(bufinfo);
	bufinfo.domain = BufferDomain::CachedHost;
	auto bitstream_cpu = device.create_buffer(bufinfo);

	bufinfo.size = encoder.get_meta_required_size();
	bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufinfo.domain = BufferDomain::Device;
	auto meta_gpu = device.create_buffer(bufinfo);
	bufinfo.domain = BufferDomain::CachedHost;
	auto meta_cpu = device.create_buffer(bufinfo);

	PyroWave::Encoder::BitstreamBuffers buffers = {};
	buffers.target_size = target_size;
	buffers.bitstream.buffer = bitstream_gpu.get();
	buffers.bitstream.size = bitstream_gpu->get_create_info().size;
	buffers.meta.buffer = meta_gpu.get();
	buffers.meta.size = meta_gpu->get_create_info().size;

	auto cmd = device.request_command_buffer(CommandBuffer::Type::AsyncCompute);
	encoder.encode(*cmd, views, buffers);

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
	cmd->copy_buffer(*bitstream_cpu, *bitstream_gpu);
	cmd->copy_buffer(*meta_cpu, *meta_gpu);
	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
				 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
	Fence fence;
	device.submit(cmd, &fence);
	fence->wait();

	std::vector<uint8_t> bitstream(target_size);
	auto *mapped_bitstream = device.map_host_buffer(*bitstream_cpu, MEMORY_ACCESS_READ_BIT);
	auto *mapped_meta = device.map_host_buffer(*meta_cpu, MEMORY_ACCESS_READ_BIT);
	Encoder::Packet packet = {};
	encoder.packetize(&packet, target_size, bitstream.data(), target_size,
					  mapped_meta, mapped_bitstream);

	cmd = device.request_command_buffer(CommandBuffer::Type::AsyncCompute);

	const Image *images[] = { &out_y.get_image(), &out_cb.get_image(), &out_cr.get_image() };

	for (auto *img : images)
	{
		cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
						   VK_PIPELINE_STAGE_NONE, VK_ACCESS_NONE,
						   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	}

	views.planes[0] = &out_y;
	views.planes[1] = &out_cb;
	views.planes[2] = &out_cr;

	decoder.clear();
	decoder.push_packet(bitstream.data() + packet.offset, packet.size);
	decoder.decode(*cmd, views);

	for (auto *img : images)
	{
		cmd->image_barrier(*img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
						   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
						   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	}

	Semaphore sem;
	device.submit(cmd, nullptr, 1, &sem);
	device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(sem), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
}

struct Sync
{
	Fence fence;
	Semaphore semaphore;
};

static Sync compute_total_errors_psnr_hvs_m(Device &device, const ImageView &a, const ImageView &b,
                                            Buffer &buffer, const float *height_factors, size_t num_height_factors,
                                            uint64_t &total_pixels)
{
	auto cmd = device.request_command_buffer();

	cmd->set_program("assets://psnr_hvs_m.comp");
	cmd->set_texture(0, 0, a);
	cmd->set_texture(0, 1, b);
	cmd->set_storage_buffer(0, 3, buffer);

	constexpr uint32_t Stride = 4;

	uint32_t last_block_x = (a.get_view_width() - 1) / Stride;
	uint32_t last_block_y = (a.get_view_height() - 1) / Stride;
	uint32_t num_blocks_x = last_block_x + 1;
	uint32_t num_blocks_y = last_block_y + 1;

	total_pixels += num_blocks_x * num_blocks_y * 64;

	uvec3 push;
	push.x = num_blocks_x;
	push.y = num_blocks_y;

	auto start_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	for (size_t i = 0; i < num_height_factors; i++)
	{
		push.z = uint32_t(i);
		cmd->push_constants(&push, 0, sizeof(push));

		auto height_pixels = float(a.get_view_height());
		float nyquist_cpd = height_pixels * height_factors[i] * muglm::pi<float>() / 360.0f;

		float csf[8][8];
		float maskcof[8][8];

		// Modified version of PSNR-HVS-M. Use our own CSF values.
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				float H = (float(x) + 0.5f) / 8.0f;
				float V = (float(y) + 0.5f) / 8.0f;
				float cpd = std::sqrt(H * H + V * V);

				// Scale the CSF to match the original quant table.
				csf[y][x] = 2.6f * contrast_sensitivity_function(cpd * nyquist_cpd);
			}
		}

		float max_csf = 0.0f;
		for (auto &row : csf)
			for (auto &v : row)
				max_csf = std::max(max_csf, v);

		float norm_factor = 1.0f / max_csf;
		norm_factor *= norm_factor;
		for (int y = 0; y < 8; y++)
			for (int x = 0; x < 8; x++)
				maskcof[y][x] = csf[y][x] * csf[y][x] * norm_factor;
		maskcof[0][0] = 0.0f;

		struct UBO
		{
			float csf_coeffs[8][8];
			float mask_coeffs[8][8];
			float inv_mask_coeffs[8][8];
		};

		auto *ubo = cmd->allocate_typed_constant_data<UBO>(0, 2, 1);
		memcpy(ubo->csf_coeffs, csf, sizeof(csf));
		memcpy(ubo->mask_coeffs, maskcof, sizeof(maskcof));
		for (int y = 0; y < 8; y++)
			for (int x = 0; x < 8; x++)
				ubo->inv_mask_coeffs[y][x] = y || x ? 1.0f / maskcof[y][x] : 0.0f;

		cmd->set_specialization_constant_mask(1);
		cmd->set_specialization_constant(0, Stride);

		cmd->dispatch((num_blocks_x + 7) / 8, (num_blocks_y + 7) / 8, 1);
	}

	auto end_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	device.register_time_interval("GPU", std::move(start_ts), std::move(end_ts), "PSNR-HVS-M group");

	Sync sync;

	cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	device.submit(cmd, &sync.fence, 1, &sync.semaphore);
	device.next_frame_context();

	return sync;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		LOGE("Usage: pyrowave-psnr-hvs-m reference.y4m target_size\n");
		return EXIT_FAILURE;
	}

	YUV4MPEGFile a, b;

	if (!a.open_read(argv[1]))
	{
		LOGE("Failed to open %s.\n", argv[1]);
		return EXIT_FAILURE;
	}

	size_t target_size = strtoul(argv[2], nullptr, 0);
	if (target_size == 0)
	{
		LOGE("Target size must not be 0.\n");
		return EXIT_FAILURE;
	}

#if 0
	if (!b.open_read(argv[2]))
	{
		fprintf(stderr, "Failed to open %s.\n", argv[2]);
		return EXIT_FAILURE;
	}

	if (a.get_width() != b.get_width() || a.get_height() != b.get_height() ||
		a.get_format() != b.get_format())
	{
		fprintf(stderr, "Mismatch in parameters (%d, %d) != (%d, %d)\n",
				a.get_width(), a.get_height(), b.get_width(), b.get_height());
		return EXIT_FAILURE;
	}
#endif

	Global::init(Global::MANAGER_FEATURE_DEFAULT_BITS, 1);
	Filesystem::setup_default_filesystem(GRANITE_FILESYSTEM(), ASSET_DIRECTORY);

	Context::SystemHandles system_handles = {};
	system_handles.filesystem = GRANITE_FILESYSTEM();

	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context context;
	context.set_system_handles(system_handles);
	context.set_num_thread_indices(2);
	if (!context.init_instance_and_device(nullptr, 0, nullptr, 0))
		return EXIT_FAILURE;
	Device device;
	device.set_context(context);

	auto num_luma_pixels = a.get_width() * a.get_height();
	auto num_chroma_pixels = num_luma_pixels;
	if (YUV4MPEGFile::format_has_subsampling(a.get_format()))
		num_chroma_pixels /= 4;

	auto bytes_per_pixel = YUV4MPEGFile::format_to_bytes_per_component(a.get_format());

	auto luma_info = ImageCreateInfo::immutable_2d_image(
		a.get_width(), a.get_height(), bytes_per_pixel == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM);
	luma_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	luma_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	luma_info.misc = IMAGE_MISC_CONCURRENT_QUEUE_ASYNC_COMPUTE_BIT |
	                 IMAGE_MISC_CONCURRENT_QUEUE_GRAPHICS_BIT |
	                 IMAGE_MISC_CONCURRENT_QUEUE_ASYNC_TRANSFER_BIT;

	auto chroma_info = luma_info;
	if (YUV4MPEGFile::format_has_subsampling(a.get_format()))
	{
		chroma_info.width /= 2;
		chroma_info.height /= 2;
	}

	//auto has_rdoc = Device::init_renderdoc_capture();
	//if (has_rdoc)
	//	device.begin_renderdoc_capture();

	constexpr uint32_t NumHeightFactors = 32;
	float height_factors[NumHeightFactors];
	for (uint32_t i = 0; i < NumHeightFactors; i++)
		height_factors[i] = 1.0f + float(i) / 16.0f;

	BufferCreateInfo atomic_info = {};
	atomic_info.size = sizeof(uint64_t) * NumHeightFactors;
	atomic_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	atomic_info.domain = BufferDomain::LinkedDeviceHost;
	atomic_info.misc = BUFFER_MISC_ZERO_INITIALIZE_BIT;

	struct WorkItem
	{
		BufferHandle buffer;
		Fence fence;
	};
	std::vector<WorkItem> work_items;
	uint64_t total_pixels = 0;
	std::unique_ptr<uint8_t[]> dummy(new uint8_t[num_chroma_pixels * bytes_per_pixel]);

	Encoder encoder;
	Decoder decoder;

	if (!encoder.init(&device, a.get_width(), a.get_height(),
	             YUV4MPEGFile::format_has_subsampling(a.get_format())
		             ? ChromaSubsampling::Chroma420
		             : ChromaSubsampling::Chroma444))
		return EXIT_FAILURE;

	if (!decoder.init(&device, a.get_width(), a.get_height(),
	             YUV4MPEGFile::format_has_subsampling(a.get_format())
		             ? ChromaSubsampling::Chroma420
		             : ChromaSubsampling::Chroma444))
		return EXIT_FAILURE;

	for (;;)
	{
		// Avoid needing to resolve WAR hazard, make sure we never block upload or async compute queues.
		auto luma_img_a = device.create_image(luma_info);
		auto cb_img_a = device.create_image(chroma_info);
		auto cr_img_a = device.create_image(chroma_info);

		auto luma_img_b = device.create_image(luma_info);
		auto cb_img_b = device.create_image(chroma_info);
		auto cr_img_b = device.create_image(chroma_info);

		const Image *images[] = {
			luma_img_a.get(),
			cb_img_a.get(),
			cr_img_a.get(),
		};

		if (!a.begin_frame())
			break;

		auto buffer = device.create_buffer(atomic_info);
		auto cmd = device.request_command_buffer(CommandBuffer::Type::AsyncTransfer);

		void *ptr;

		cmd->begin_barrier_batch();
		for (auto *img : images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
							   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
							   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
		}
		cmd->end_barrier_batch();

		ptr = cmd->update_image(*luma_img_a, {}, {luma_img_a->get_width(), luma_img_a->get_height(), 1}, 0, 0,
		                  {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
		if (!a.read(ptr, num_luma_pixels * bytes_per_pixel))
		{
			device.submit_discard(cmd);
			break;
		}

		ptr = cmd->update_image(*cb_img_a, {}, {cb_img_a->get_width(), cb_img_a->get_height(), 1}, 0, 0,
						  {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
		if (!a.read(ptr, num_chroma_pixels * bytes_per_pixel))
		{
			device.submit_discard(cmd);
			break;
		}

		ptr = cmd->update_image(*cr_img_a, {}, {cr_img_a->get_width(), cr_img_a->get_height(), 1}, 0, 0,
						  {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
		if (!a.read(ptr, num_chroma_pixels * bytes_per_pixel))
		{
			device.submit_discard(cmd);
			break;
		}

		cmd->begin_barrier_batch();
		for (auto *img : images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
							   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
							   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}
		cmd->end_barrier_batch();

		Semaphore sem[2];
		device.submit(cmd, nullptr, 2, sem);
		device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(sem[0]), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
		device.add_wait_semaphore(CommandBuffer::Type::AsyncCompute, std::move(sem[1]), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);

		roundtrip_pyrowave(device, encoder, decoder,
		                   luma_img_b->get_view(), cb_img_b->get_view(), cr_img_b->get_view(),
		                   luma_img_a->get_view(), cb_img_a->get_view(), cr_img_a->get_view(),
		                   target_size);

		WorkItem item;
		auto sync = compute_total_errors_psnr_hvs_m(device, luma_img_a->get_view(), luma_img_b->get_view(),
		                                            *buffer, height_factors, NumHeightFactors, total_pixels);
		item.fence = std::move(sync.fence);
		item.buffer = std::move(buffer);
		work_items.push_back(std::move(item));
	}

	//if (has_rdoc)
	//	device.end_renderdoc_capture();

	double total_square_error_per_height_factor[NumHeightFactors] = {};

	for (auto &item : work_items)
	{
		item.fence->wait();
		auto *ptr = static_cast<const uint64_t *>(device.map_host_buffer(*item.buffer, MEMORY_ACCESS_READ_BIT));
		for (uint32_t i = 0; i < NumHeightFactors; i++)
			total_square_error_per_height_factor[i] += ldexp(double(ptr[i]), -24);
	}

	for (uint32_t i = 0; i < NumHeightFactors; i++)
	{
		total_square_error_per_height_factor[i] /= double(total_pixels);
		double peak_signal = a.is_full_range() ? 1.0f * 1.0f : (223.0f * 223.0f) / (255.0f * 255.0f);
		LOGI("Resolution (%u x %u), TargetSize %zu || HeightFactor = %.2f || PSNR-HVS-M-H: (Y) %4.4f dB\n",
		     a.get_width(), a.get_height(), target_size,
		     height_factors[i],
		     10.0 * std::log10(peak_signal / total_square_error_per_height_factor[i]));
	}
}