// Copyright (c) 2025-2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

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
#include "cli_parser.hpp"
#include "ffmpeg_decode.hpp"
#include "thread_group.hpp"
#include "path_utils.hpp"

using namespace Vulkan;
using namespace Granite;
using namespace PyroWave;
using namespace Util;

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

	Encoder::BitstreamBuffers buffers = {};
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

	return sync;
}

static void print_help()
{
	LOGE("pyrowave-psnr-hvs-m\n"
		"\t[--help]\n"
		"\t[--reference <path>]\n"
		"\t[--pyrowave-target-size <bytes>]\n"
		"\t[--pyrowave-target-size-range <start> <end> <step>]\n"
		"\t[--distorted <path>]\n");
}

struct WorkItem
{
	BufferHandle buffer;
	Fence fence;
};

static constexpr uint32_t NumHeightFactors = 16;

static void compute_psnr_hvs_m(double (&psnr)[NumHeightFactors], Device &device, WorkItem *items, size_t num_items, uint64_t total_pixels, bool full_range)
{
	double total_square_error_per_height_factor[NumHeightFactors] = {};

	for (size_t index = 0; index < num_items; index++)
	{
		auto &item = items[index];
		item.fence->wait();
		auto *ptr = static_cast<const uint64_t *>(device.map_host_buffer(*item.buffer, MEMORY_ACCESS_READ_BIT));
		for (uint32_t i = 0; i < NumHeightFactors; i++)
			total_square_error_per_height_factor[i] += ldexp(double(ptr[i]), -24);
	}

	const double peak_signal = full_range ? 1.0f * 1.0f : (223.0f * 223.0f) / (255.0f * 255.0f);

	for (uint32_t i = 0; i < NumHeightFactors; i++)
		psnr[i] = 10.0 * std::log10(double(total_pixels) * peak_signal / total_square_error_per_height_factor[i]);
}

struct PSNRTestCase
{
	std::string desc;
	std::unique_ptr<VideoDecoder> decoder;
	size_t pyrowave_size = 0;
	uint64_t total_pixels = 0;
	double psnr_hvs_m[NumHeightFactors] = {};
	std::vector<WorkItem> work_items;
};

struct Reference
{
	std::unique_ptr<VideoDecoder> decoder;
	unsigned frame_count = 0;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t num_planes = 0;
	bool chroma_subsample = false;
	bool full_range = true; // TODO: Assume for now.
	VkFormat luma_format = VK_FORMAT_UNDEFINED;
	VkFormat chroma_format = VK_FORMAT_UNDEFINED;
	std::string desc;
};

struct PyroWaveRoundtripper
{
	std::unique_ptr<Encoder> encoder;
	std::unique_ptr<Decoder> decoder;
	uint32_t width = 0;
	uint32_t height = 0;
	ChromaSubsampling chroma = {};

	bool ensure(Device &device, uint32_t width_, uint32_t height_, ChromaSubsampling chroma_)
	{
		if (width == width_ && height == height_ && chroma == chroma_)
			return true;

		width = width_;
		height = height_;
		chroma = chroma_;

		encoder = std::make_unique<Encoder>();
		decoder = std::make_unique<Decoder>();
		if (!encoder->init(&device, width, height, chroma))
			return false;
		if (!decoder->init(&device, width, height, chroma))
			return false;
		return true;
	}
};

static float get_height_factor_from_index(uint32_t index)
{
	return 1.0f + float(index) / 8.0f;
}

int main(int argc, char **argv)
{
	std::vector<size_t> pyrowave_sizes;
	std::vector<std::string> distorted;
	std::vector<std::string> reference_paths;
	CLICallbacks cbs;

	cbs.add("--help", [&](CLIParser &parser) { parser.end(); });
	cbs.add("--reference", [&](CLIParser &parser) { reference_paths.emplace_back(parser.next_string()); });
	cbs.add("--pyrowave-target-size", [&](CLIParser &parser) { pyrowave_sizes.push_back(parser.next_uint()); });
	cbs.add("--pyrowave-target-size-range", [&](CLIParser &parser)
	{
		uint32_t start_size = parser.next_uint();
		uint32_t end_size = parser.next_uint();
		uint32_t step_size = parser.next_uint();
		if (step_size == 0)
			throw std::invalid_argument("step size cannot be 0.");

		while (start_size <= end_size)
		{
			pyrowave_sizes.push_back(start_size);
			start_size += step_size;
		}
	});
	cbs.add("--distorted", [&](CLIParser &parser) { distorted.emplace_back(parser.next_string()); });

	CLIParser parser(std::move(cbs), argc - 1, argv + 1);
	if (!parser.parse())
	{
		print_help();
		return EXIT_FAILURE;
	}
	else if (parser.is_ended_state())
	{
		print_help();
		return EXIT_SUCCESS;
	}

	if (reference_paths.empty())
	{
		LOGE("Need to provide --reference\n");
		print_help();
		return EXIT_SUCCESS;
	}

	if (reference_paths.size() > 1 && !distorted.empty())
	{
		LOGE("When using external --distorted files, only one reference can be used.\n");
		print_help();
		return EXIT_SUCCESS;
	}

	if (pyrowave_sizes.empty() && distorted.empty())
	{
		LOGE("Need to provide --distorted or --pyrowave-target-size at least once\n");
		print_help();
		return EXIT_SUCCESS;
	}

	std::vector<Reference> references;
	VideoDecoder::DecodeOptions decode_options = {};
	std::vector<PSNRTestCase> test_cases;
	decode_options.blocking = true;
	// Workaround buggy FFmpeg with Vulkan FFV1 decode.
	// Just instantly faults my GPU.
	decode_options.hwdevice = "none";
	decode_options.threads = std::min<uint32_t>(16u, std::thread::hardware_concurrency());

	for (auto &ref : reference_paths)
	{
		Reference reference;
		reference.decoder = std::make_unique<VideoDecoder>();
		reference.desc = Path::basename(ref);

		if (!reference.decoder->init(nullptr, ref.c_str(), decode_options))
		{
			LOGE("Failed to open reference \"%s\"\n", ref.c_str());
			return EXIT_FAILURE;
		}

		references.push_back(std::move(reference));
	}

	decode_options.hwdevice = nullptr;

	for (auto &pyro : pyrowave_sizes)
	{
		PSNRTestCase test_case;
		test_case.desc = "pyrowave_" + std::to_string(pyro);
		test_case.pyrowave_size = pyro;
		test_cases.push_back(std::move(test_case));
	}

	for (auto &dist : distorted)
	{
		PSNRTestCase test_case;
		test_case.desc = dist;
		test_case.decoder = std::make_unique<VideoDecoder>();
		if (!test_case.decoder->init(nullptr, dist.c_str(), decode_options))
		{
			LOGE("Failed to open test case path: \"%s\"\n", dist.c_str());
			return EXIT_FAILURE;
		}
		test_cases.push_back(std::move(test_case));
	}

	Global::init(Global::MANAGER_FEATURE_DEFAULT_BITS, 1);
	Filesystem::setup_default_filesystem(GRANITE_FILESYSTEM(), ASSET_DIRECTORY);

	Context::SystemHandles system_handles = {};
	system_handles.filesystem = GRANITE_FILESYSTEM();
	system_handles.thread_group = GRANITE_THREAD_GROUP();

	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context context;
	context.set_system_handles(system_handles);
	context.set_num_thread_indices(GRANITE_THREAD_GROUP()->get_num_threads() + 1);
	if (!context.init_instance_and_device(nullptr, 0, nullptr, 0))
		return EXIT_FAILURE;

	Device device;
	device.set_context(context);

	FFmpegDecode::Shaders<> shaders;
	auto *comp = device.get_shader_manager().register_compute("builtin://shaders/util/yuv_to_rgb.comp");
	shaders.yuv_to_rgb = comp->register_variant({})->get_program();

	for (auto &test_case : test_cases)
	{
		if (test_case.decoder)
		{
			if (!test_case.decoder->begin_device_context(&device, shaders))
				return EXIT_FAILURE;

			if (!test_case.decoder->play())
			{
				LOGE("Failed to start payback of \"%s\".\n", test_case.desc.c_str());
				return EXIT_FAILURE;
			}
		}
	}

	auto has_rdoc = Device::init_renderdoc_capture();

	float height_factors[NumHeightFactors];
	for (uint32_t i = 0; i < NumHeightFactors; i++)
		height_factors[i] = get_height_factor_from_index(i);

	BufferCreateInfo atomic_info = {};
	atomic_info.size = sizeof(uint64_t) * NumHeightFactors;
	atomic_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	atomic_info.domain = BufferDomain::LinkedDeviceHost;
	atomic_info.misc = BUFFER_MISC_ZERO_INITIALIZE_BIT;

	PyroWaveRoundtripper pyrowave;

	struct
	{
		Semaphore timeline;
		uint64_t timeline_value = 0;
	} graphics_timeline, compute_timeline;

	graphics_timeline.timeline = device.request_semaphore(VK_SEMAPHORE_TYPE_TIMELINE);
	compute_timeline.timeline = device.request_semaphore(VK_SEMAPHORE_TYPE_TIMELINE);

	for (auto &test_case : test_cases)
	{
		if (!test_case.decoder)
			continue;

		// Throw away the first frame, with predictive codecs the first frame may be more damaged than usual.
		VideoFrame frame = {};
		if (!test_case.decoder->acquire_video_frame(frame))
		{
			LOGE("Failed to acquire first frame.\n");
			return EXIT_FAILURE;
		}
		test_case.decoder->release_video_frame(frame.index, std::move(frame.sem));
	}

	constexpr uint32_t CaptureFrameStart = 10;
	constexpr uint32_t CaptureFrameEnd = 10;

	for (auto &reference : references)
	{
		if (!reference.decoder->begin_device_context(&device, shaders))
			return EXIT_FAILURE;
		if (!reference.decoder->play())
		{
			LOGE("Failed to start payback of reference.\n");
			return EXIT_FAILURE;
		}

		// Throw away the first frame, with predictive codecs the first frame may be more damaged than usual.
		VideoFrame frame;
		if (!reference.decoder->acquire_video_frame(frame))
		{
			LOGE("Failed to acquire first frame.\n");
			return EXIT_FAILURE;
		}
		reference.decoder->release_video_frame(frame.index, std::move(frame.sem));

		reference.width = frame.view->get_view_width();
		reference.height = frame.view->get_view_height();

		switch (frame.view->get_format())
		{
		case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
			reference.luma_format = VK_FORMAT_R8_UNORM;
			reference.chroma_format = VK_FORMAT_R8G8_UNORM;
			reference.chroma_subsample = true;
			reference.num_planes = 2;
			break;

		case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
			reference.luma_format = VK_FORMAT_R8_UNORM;
			reference.chroma_format = VK_FORMAT_R8_UNORM;
			reference.chroma_subsample = true;
			reference.num_planes = 3;
			break;

		case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM:
			reference.luma_format = VK_FORMAT_R8_UNORM;
			reference.chroma_format = VK_FORMAT_R8G8_UNORM;
			reference.chroma_subsample = false;
			reference.num_planes = 2;
			break;

		case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM:
			reference.luma_format = VK_FORMAT_R8_UNORM;
			reference.chroma_format = VK_FORMAT_R8_UNORM;
			reference.chroma_subsample = false;
			reference.num_planes = 3;
			break;

		default:
			LOGE("TODO: Add more format support\n");
			return EXIT_FAILURE;
		}

		for (;;)
		{
			if (has_rdoc && reference.frame_count == CaptureFrameStart)
				device.begin_renderdoc_capture();

			VideoFrame reference_frame = {};
			if (!reference.decoder->acquire_video_frame(reference_frame))
				break;

			device.add_wait_semaphore(CommandBuffer::Type::AsyncCompute, std::move(reference_frame.sem),
									  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);

			if (!distorted.empty())
			{
				// Dumb workaround so that we can block both queues.
				auto binary = device.request_timeline_semaphore_as_binary(
					*compute_timeline.timeline, ++compute_timeline.timeline_value);
				device.submit_empty(CommandBuffer::Type::AsyncCompute, nullptr, binary.get());
				device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(binary), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
			}

			ImageViewHandle reference_views[3];
			{
				ImageViewCreateInfo view_info = {};
				view_info.image = &reference_frame.view->get_image();
				view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
				view_info.format = reference.luma_format;
				view_info.aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;

				for (int i = 0; i < 3; i++)
				{
					view_info.format = i ? reference.chroma_format : reference.luma_format;

					if (reference.num_planes == 2 && i == 2)
					{
						view_info.aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
						view_info.swizzle.r = VK_COMPONENT_SWIZZLE_G;
					}
					else
					{
						view_info.aspect = VK_IMAGE_ASPECT_PLANE_0_BIT << i;
						view_info.swizzle.r = VK_COMPONENT_SWIZZLE_IDENTITY;
					}

					reference_views[i] = device.create_image_view(view_info);
				}
			}

			for (auto &test_case : test_cases)
			{
				const ImageView *psnr_test_view = nullptr;
				ImageViewHandle luma_test_view;
				VideoFrame test_frame = {};
				ImageHandle plane_images[3];

				auto work_buffer = device.create_buffer(atomic_info);

				if (test_case.decoder)
				{
					if (!test_case.decoder->acquire_video_frame(test_frame))
						continue;

					device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(test_frame.sem),
											  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);

					ImageViewCreateInfo view_info = {};
					view_info.image = &test_frame.view->get_image();
					view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
					view_info.format = reference.luma_format;
					view_info.aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
					luma_test_view = device.create_image_view(view_info);
					psnr_test_view = luma_test_view.get();

					if (test_frame.pts < 0.0 || muglm::abs(test_frame.pts - reference_frame.pts) > 0.001)
					{
						LOGI("Test %s, frame count %u, reference PTS %.3f != test PTS %.3f\n",
							 test_case.desc.c_str(), reference.frame_count,
							 reference_frame.pts, test_frame.pts);
					}
				}
				else
				{
					if (!pyrowave.ensure(device, reference.width, reference.height,
										 reference.chroma_subsample ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
						return EXIT_FAILURE;

					auto image_info = ImageCreateInfo::immutable_2d_image(reference.width, reference.height, reference.luma_format);
					image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
					image_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

					plane_images[0] = device.create_image(image_info);
					if (reference.chroma_subsample)
					{
						image_info.width /= 2;
						image_info.height /= 2;
					}
					plane_images[1] = device.create_image(image_info);
					plane_images[2] = device.create_image(image_info);

					roundtrip_pyrowave(device, *pyrowave.encoder, *pyrowave.decoder,
									   plane_images[0]->get_view(), plane_images[1]->get_view(),
									   plane_images[2]->get_view(),
									   *reference_views[0], *reference_views[1], *reference_views[2],
									   test_case.pyrowave_size);

					psnr_test_view = &plane_images[0]->get_view();
				}

				WorkItem item;
				auto sync = compute_total_errors_psnr_hvs_m(
					device, *reference_views[0], *psnr_test_view,
					*work_buffer, height_factors, NumHeightFactors, test_case.total_pixels);
				item.fence = std::move(sync.fence);
				item.buffer = std::move(work_buffer);
				test_case.work_items.push_back(std::move(item));

				if (test_case.decoder)
				{
					auto binary = device.request_timeline_semaphore_as_binary(*graphics_timeline.timeline, ++graphics_timeline.timeline_value);
					device.submit_empty(CommandBuffer::Type::Generic, nullptr, binary.get());
					test_case.decoder->release_video_frame(test_frame.index, std::move(binary));
				}
			}

			// Release the reference frame.
			{
				auto binary = device.request_timeline_semaphore_as_binary(*graphics_timeline.timeline, ++graphics_timeline.timeline_value);
				device.submit_empty(CommandBuffer::Type::Generic, nullptr, binary.get());
				reference.decoder->release_video_frame(reference_frame.index, std::move(binary));
			}

			if (has_rdoc && reference.frame_count == CaptureFrameEnd)
			{
				device.end_renderdoc_capture();
				break;
			}

			reference.frame_count++;
			LOGI("Completed %u frames of %s ...\n", reference.frame_count, reference.desc.c_str());

			device.next_frame_context();
		}

		// Save some resources.
		reference = {};
	}

	for (auto &test_case : test_cases)
	{
		compute_psnr_hvs_m(test_case.psnr_hvs_m, device, test_case.work_items.data(), test_case.work_items.size(),
		                   test_case.total_pixels, true /* full_range */);

		for (uint32_t i = 0; i < NumHeightFactors; i++)
		{
			LOGI("Test: %s || TargetSize %zu || HeightFactor = %.2f || PSNR-HVS-M-H: (Y) %4.4f dB\n",
				test_case.desc.c_str(), test_case.pyrowave_size,
				height_factors[i], test_case.psnr_hvs_m[i]);
		}
	}

	test_cases.clear();
	references.clear();
	graphics_timeline = {};
	compute_timeline = {};
}