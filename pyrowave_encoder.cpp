// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#include "pyrowave_encoder.hpp"
#include "device.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "math.hpp"
#include "pyrowave_common.hpp"
#include <algorithm>
#include <cmath>

namespace PyroWave
{
using namespace Granite;
using namespace Vulkan;

static constexpr int BlockSpaceSubdivision = 16;
static constexpr int NumRDOBuckets = 128;
static constexpr int RDOBucketOffset = 64;

static int compute_block_count_per_subdivision(int num_blocks)
{
	int per_subdivision = align(num_blocks, BlockSpaceSubdivision) / BlockSpaceSubdivision;
	per_subdivision = int(Util::next_pow2(per_subdivision));
	return per_subdivision;
}

struct QuantizerPushData
{
	ivec2 resolution;
	vec2 inv_resolution;
	float input_layer;
	float quant_resolution;
	int32_t block_offset;
	int32_t block_stride;
};

struct BlockPackingPushData
{
	ivec2 resolution;
	ivec2 resolution_64x64_blocks;
	ivec2 resolution_16x16_blocks;
	uint32_t quant_resolution_code;
	uint32_t sequence_count;
	uint32_t block_offset_64x64;
	uint32_t block_stride_64x64;
	uint32_t block_offset_16x16;
	uint32_t block_stride_16x16;
};

struct AnalyzeRateControlPushData
{
	ivec2 resolution;
	ivec2 resolution_16x16_blocks;
	float step_size;
	float rdo_distortion_scale;
	int32_t block_offset_16x16;
	int32_t block_stride_16x16;
	int32_t block_offset_64x64;
	int32_t block_stride_64x64;
	uint32_t total_wg_count;
	uint32_t num_blocks_aligned;
	uint32_t block_index_shamt;
};

struct RDOperation
{
	int32_t quant;
	uint16_t block_offset;
	uint16_t block_saving;
};

struct Encoder::Impl : public WaveletBuffers
{
	BufferHandle bucket_buffer, meta_buffer, deadzone_buffer, payload_data, quant_buffer;

	bool encode(CommandBuffer &cmd, const ViewBuffers &views, const BitstreamBuffers &buffers);

	bool dwt(CommandBuffer &cmd, const ViewBuffers &views);
	bool quant(CommandBuffer &cmd);
	bool analyze_rdo(CommandBuffer &cmd);
	bool resolve_rdo(CommandBuffer &cmd, size_t target_payload_size);
	bool block_packing(CommandBuffer &cmd, const BitstreamBuffers &buffers);

	float get_noise_power_normalized_quant_resolution(int level, int component, int band) const;
	float get_quant_resolution(int level, int component, int band) const;
	float get_quant_rdo_distortion_scale(int level, int component, int band) const;

	void init_block_meta() override;

	size_t compute_num_packets(const void *meta, size_t packet_boundary) const;

	size_t packetize(Packet *packets, size_t packet_boundary,
	                 void *bitstream, size_t size,
	                 const void *mapped_meta, const void *mapped_bitstream) const;

	void report_stats(const void *mapped_meta, const void *mapped_bitstream) const;
	void analyze_alternative_packing(const void *mapped_meta, const void *mapped_bitstream) const;

	bool validate_bitstream(const uint32_t *bitstream, const BitstreamPacket *meta, uint32_t block_index) const;

	uint32_t sequence_count = 0;
};

float Encoder::Impl::get_quant_rdo_distortion_scale(int level, int component, int band) const
{
	// From my Linelet master thesis. Copy paste 11 years later, ah yes :D
	float horiz_midpoint = (band & 1) ? 0.75f : 0.25f;
	float vert_midpoint = (band & 2) ? 0.75f : 0.25f;

	// Normal PC monitors.
	constexpr float dpi = 96.0f;
	// Compromise between couch gaming and desktop.
	constexpr float viewing_distance = 1.0f;
	constexpr float cpd_nyquist = 0.34f * viewing_distance * dpi;

	float cpd = std::sqrt(horiz_midpoint * horiz_midpoint + vert_midpoint * vert_midpoint) *
	                  cpd_nyquist * std::exp2(-float(level));

	// Don't allow a situation where we're quantizing LL band hard.
	cpd = std::max(cpd, 8.0f);

	float csf = 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));

	// Heavily discount chroma quality.
	if (component != 0 && level != DecompositionLevels - 1)
		csf *= 0.4f;

	// Due to filtering, distortion in lower bands will result in more noise power.
	// By scaling the distortion by this factor, we ensure uniform results.
	float resolution = get_noise_power_normalized_quant_resolution(level, component, band);
	float weighted_resolution = csf * resolution;

	// The distortion is scaled in terms of power, not amplitude.
	return weighted_resolution * weighted_resolution;
}

float Encoder::Impl::get_quant_resolution(int level, int component, int band) const
{
	// FP16 range is limited, and this is more than a good enough initial estimate.
	return std::min<float>(512.0f, get_noise_power_normalized_quant_resolution(level, component, band));
}

float Encoder::Impl::get_noise_power_normalized_quant_resolution(int level, int component, int band) const
{
	// The initial quantization resolution aims for a flat spectrum with noise power normalization.
	// The low-pass gain for CDF 9/7 is 6 dB (1 bit). Every decomposition level subtracts 6 dB.

	// Maybe make this based on the max rate to have a decent initial estimate.
	int bits = 6;

	if (band == 0)
		bits += 2;
	else if (band < 3)
		bits += 1;

	bits += level;

	// Chroma starts at level 1, subtract one bit.
	if (component != 0)
		bits--;

	return float(1 << bits);
}

void Encoder::Impl::init_block_meta()
{
	WaveletBuffers::init_block_meta();

	BufferCreateInfo info;
	info.domain = BufferDomain::Device;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	info.size = block_count_16x16 * sizeof(DeadZone);
	deadzone_buffer = device->create_buffer(info);
	device->set_name(*deadzone_buffer, "deadzone-buffer");

	info.size = block_count_16x16 * sizeof(BlockMeta);
	meta_buffer = device->create_buffer(info);
	device->set_name(*meta_buffer, "meta-buffer");

	// Worst case estimate.
	info.size = aligned_width * aligned_height * 2;
	payload_data = device->create_buffer(info);
	device->set_name(*payload_data, "payload-data");

	info.size = block_count_64x64 * sizeof(uint32_t);
	quant_buffer = device->create_buffer(info);
	device->set_name(*quant_buffer, "quant-buffer");

	info.size = RDOBucketOffset;
	info.size += NumRDOBuckets * BlockSpaceSubdivision * sizeof(uint32_t);
	info.size += NumRDOBuckets * compute_block_count_per_subdivision(block_count_64x64) * BlockSpaceSubdivision * sizeof(RDOperation);
	bucket_buffer = device->create_buffer(info);
	device->set_name(*bucket_buffer, "bucket-buffer");
}

bool Encoder::Impl::block_packing(CommandBuffer &cmd, const BitstreamBuffers &buffers)
{
	cmd.begin_region("DWT block packing");
	auto start_packing = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	cmd.set_program(shaders.block_packing);
	cmd.set_storage_buffer(0, 0, *buffers.bitstream.buffer, buffers.bitstream.offset, buffers.bitstream.size);
	cmd.set_storage_buffer(0, 1, *buffers.meta.buffer, buffers.meta.offset, buffers.meta.size);
	cmd.set_storage_buffer(0, 2, *meta_buffer);
	cmd.set_storage_buffer(0, 3, *payload_data);
	cmd.set_storage_buffer(0, 4, *deadzone_buffer);
	cmd.set_storage_buffer(0, 5, *quant_buffer);

	if (device->supports_subgroup_size_log2(true, 4, 6))
	{
		cmd.set_subgroup_size_log2(true, 4, 6);
	}
	else
	{
		LOGI("No compatible subgroup size config.\n");
		return false;
	}

	for (int level = 0; level < DecompositionLevels; level++)
	{
		auto level_width = wavelet_img->get_width(level);
		auto level_height = wavelet_img->get_height(level);

		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			char label[128];
			snprintf(label, sizeof(label), "level %d, component %d", level, component);
			cmd.begin_region(label);

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				BlockPackingPushData packing_push = {};
				packing_push.resolution = ivec2(level_width, level_height);
				packing_push.resolution_64x64_blocks = ivec2((level_width + 63) / 64, (level_height + 63) / 64);
				packing_push.resolution_16x16_blocks = ivec2((level_width + 15) / 16, (level_height + 15) / 16);

				auto quant_res = get_quant_resolution(level, component, band);
				packing_push.quant_resolution_code = encode_quant(1.0f / quant_res);
				packing_push.sequence_count = sequence_count;

				auto &meta = block_meta[component][level][band];

				packing_push.block_offset_64x64 = meta.block_offset_64x64;
				packing_push.block_stride_64x64 = meta.block_stride_64x64;
				packing_push.block_offset_16x16 = meta.block_offset_16x16;
				packing_push.block_stride_16x16 = meta.block_stride_16x16;
				cmd.push_constants(&packing_push, 0, sizeof(packing_push));

				cmd.dispatch((packing_push.resolution_64x64_blocks.x + 1) / 2,
				             (packing_push.resolution_64x64_blocks.y + 1) / 2,
				             1);
			}

			cmd.end_region();
		}
	}

	auto end_packing = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
	            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);

	device->register_time_interval("GPU", std::move(start_packing), std::move(end_packing), "Packing");
	cmd.end_region();

	return true;
}

bool Encoder::Impl::resolve_rdo(CommandBuffer &cmd, size_t target_payload_size)
{
	cmd.begin_region("DWT resolve");

	auto start_resolve = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	if (target_payload_size >= sizeof(BitstreamSequenceHeader))
		target_payload_size -= sizeof(BitstreamSequenceHeader);

	cmd.set_specialization_constant_mask(1);

	if (device->supports_subgroup_size_log2(true, 6, 6))
	{
		cmd.set_specialization_constant(0, 64);
		cmd.set_subgroup_size_log2(true, 6, 6);
	}
	else if (device->supports_subgroup_size_log2(true, 4, 4))
	{
		cmd.set_specialization_constant(0, 16);
		cmd.set_subgroup_size_log2(true, 4, 4);
	}
	else if (device->supports_subgroup_size_log2(true, 5, 5))
	{
		cmd.set_specialization_constant(0, 32);
		cmd.set_subgroup_size_log2(true, 5, 5);
	}
	else
	{
		LOGI("No compatible subgroup size config.\n");
		return false;
	}

	cmd.set_program(shaders.resolve_rate_control);

	struct
	{
		uint32_t target_payload_size;
		uint32_t num_blocks_per_subdivision;
	} push = {};

	push.target_payload_size = target_payload_size / sizeof(uint32_t);
	push.num_blocks_per_subdivision = compute_block_count_per_subdivision(block_count_64x64);
	cmd.push_constants(&push, 0, sizeof(push));
	cmd.set_storage_buffer(0, 0, *bucket_buffer);
	cmd.set_storage_buffer(0, 1, *quant_buffer);
	cmd.dispatch(NumRDOBuckets * BlockSpaceSubdivision, 1, 1);

	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
	cmd.end_region();

	auto end_resolve = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	device->register_time_interval("GPU", std::move(start_resolve), std::move(end_resolve), "Resolve");
	cmd.set_specialization_constant_mask(0);
	return true;
}

bool Encoder::Impl::analyze_rdo(CommandBuffer &cmd)
{
	auto start_analyze = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	cmd.begin_region("DWT analyze");
	cmd.set_program(shaders.analyze_rate_control);

	if (device->supports_subgroup_size_log2(true, 4, 6))
	{
		cmd.set_subgroup_size_log2(true, 4, 6);
	}
	else
	{
		LOGI("No compatible subgroup size config.\n");
		return false;
	}

	// Quantize
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			AnalyzeRateControlPushData push = {};

			char label[128];
			snprintf(label, sizeof(label), "level %d, component %d", level, component);
			cmd.begin_region(label);

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				auto level_width = wavelet_img->get_width(level);
				auto level_height  = wavelet_img->get_height(level);

				float quant_res = get_quant_resolution(level, component, band);

				push.resolution.x = level_width;
				push.resolution.y = level_height;
				push.resolution_16x16_blocks.x = (level_width + 15) / 16;
				push.resolution_16x16_blocks.y = (level_height + 15) / 16;
				push.step_size = decode_quant(encode_quant(1.0f / quant_res));
				push.rdo_distortion_scale = get_quant_rdo_distortion_scale(level, component, band);
				push.block_offset_16x16 = block_meta[component][level][band].block_offset_16x16;
				push.block_stride_16x16 = block_meta[component][level][band].block_stride_16x16;
				push.block_offset_64x64 = block_meta[component][level][band].block_offset_64x64;
				push.block_stride_64x64 = block_meta[component][level][band].block_stride_64x64;
				push.total_wg_count = block_count_64x64;
				push.num_blocks_aligned = compute_block_count_per_subdivision(block_count_64x64) * BlockSpaceSubdivision;
				push.block_index_shamt = Util::floor_log2(compute_block_count_per_subdivision(block_count_64x64));

				cmd.push_constants(&push, 0, sizeof(push));

				cmd.set_storage_buffer(0, 0, *bucket_buffer);
				cmd.set_storage_buffer(0, 1, *meta_buffer);
				cmd.set_storage_buffer(0, 2, *deadzone_buffer);

				cmd.dispatch((level_width + 63) / 64, (level_height + 63) / 64, 1);
			}

			cmd.end_region();
		}
	}

	cmd.end_region();
	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

	auto end_analyze = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	device->register_time_interval("GPU", std::move(start_analyze), std::move(end_analyze), "Analyze");
	return true;
}

bool Encoder::Impl::quant(CommandBuffer &cmd)
{
	auto start_quant = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	cmd.begin_region("DWT quantize");
	cmd.set_program(shaders.wavelet_quant);

	cmd.set_specialization_constant_mask(1);
	if (device->supports_subgroup_size_log2(true, 6, 6))
	{
		cmd.set_specialization_constant(0, 64);
		cmd.set_subgroup_size_log2(true, 6, 6);
	}
#if 0
	// Disgustingly slow path for some reason.
	else if (device->supports_subgroup_size_log2(true, 3, 4))
	{
		cmd.set_specialization_constant(0, 256);
		cmd.set_subgroup_size_log2(true, 3, 4);
	}
#endif
	else if (device->supports_subgroup_size_log2(true, 4, 4))
	{
		cmd.set_specialization_constant(0, 16);
		cmd.set_subgroup_size_log2(true, 4, 4);
	}
	else if (device->supports_subgroup_size_log2(true, 5, 5))
	{
		cmd.set_specialization_constant(0, 32);
		cmd.set_subgroup_size_log2(true, 5, 5);
	}
	else
	{
		LOGI("No compatible subgroup size config.\n");
		return false;
	}

	// Quantize
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			QuantizerPushData push = {};

			char label[128];
			snprintf(label, sizeof(label), "DWT quant, level %d, component %d", level, component);
			cmd.begin_region(label);

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				float quant_res = get_quant_resolution(level, component, band);

				push.resolution.x = wavelet_img->get_width(level);
				push.resolution.y = wavelet_img->get_height(level);
				push.inv_resolution.x = 1.0f / float(push.resolution.x);
				push.inv_resolution.y = 1.0f / float(push.resolution.y);
				push.input_layer = float(band);
				push.quant_resolution = 1.0f / decode_quant(encode_quant(1.0f / quant_res));

				int blocks_x = (push.resolution.x + 15) / 16;
				int blocks_y = (push.resolution.y + 15) / 16;

				push.block_offset = block_meta[component][level][band].block_offset_16x16;
				push.block_stride = block_meta[component][level][band].block_stride_16x16;

				cmd.push_constants(&push, 0, sizeof(push));

				cmd.set_texture(0, 0, *component_layer_views[component][level], *border_sampler);
				cmd.set_storage_buffer(0, 1, *meta_buffer);
				cmd.set_storage_buffer(0, 2, *deadzone_buffer);
				cmd.set_storage_buffer(0, 3, *payload_data);

				cmd.dispatch(blocks_x, blocks_y, 1);
			}

			cmd.end_region();
		}
	}

	cmd.end_region();
	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

	auto end_quant = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	device->register_time_interval("GPU", std::move(start_quant), std::move(end_quant), "Quant");
	cmd.set_specialization_constant_mask(0);
	return true;
}

bool Encoder::Impl::dwt(CommandBuffer &cmd, const ViewBuffers &views)
{
	struct Push
	{
		uvec2 resolution;
		vec2 inv_resolution;
		uvec2 aligned_resolution;
	} push = {};

	// Forward transforms.
	cmd.set_program(shaders.dwt);

	// Only need simple 2-lane swaps.
	cmd.set_subgroup_size_log2(true, 2, 7);
	cmd.set_specialization_constant_mask(1);
	cmd.set_specialization_constant(0, false);

	auto start_dwt = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	for (int output_level = 0; output_level < DecompositionLevels; output_level++)
	{
		if (output_level > 0)
		{
			push.resolution = uvec2(component_ll_views[0][output_level - 1]->get_view_width(),
			                        component_ll_views[0][output_level - 1]->get_view_height());
			push.aligned_resolution = push.resolution;
		}
		else
		{
			push.resolution = uvec2(views.planes[0]->get_view_width(), views.planes[0]->get_view_height());
			push.aligned_resolution.x = aligned_width;
			push.aligned_resolution.y = aligned_height;
		}

		push.inv_resolution.x = 1.0f / float(push.resolution.x);
		push.inv_resolution.y = 1.0f / float(push.resolution.y);
		cmd.push_constants(&push, 0, sizeof(push));

		if (output_level == 0)
		{
			cmd.set_specialization_constant(0, /*mode == Mode::YCbCr_420*/ true);
			cmd.set_texture(0, 0, *views.planes[0], *mirror_repeat_sampler);
			cmd.set_storage_texture(0, 1, *component_layer_views[0][output_level]);

			//if (mode == Mode::RGB)
			//	cmd.begin_region("DWT RGB -> YCbCr");
			//else
			{
				cmd.begin_region("DWT level 0 Y");
			}

			cmd.dispatch((push.aligned_resolution.x + 31) / 32, (push.aligned_resolution.y + 31) / 32, 1);
			cmd.end_region();
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
				if (c != 0 && output_level == 1)
				{
					push.resolution = uvec2(views.planes[c]->get_view_width(), views.planes[c]->get_view_height());
					push.aligned_resolution.x = aligned_width >> output_level;
					push.aligned_resolution.y = aligned_height >> output_level;
					push.inv_resolution.x = 1.0f / float(push.resolution.x);
					push.inv_resolution.y = 1.0f / float(push.resolution.y);
					cmd.push_constants(&push, 0, sizeof(push));
					cmd.set_texture(0, 0, *views.planes[c], *mirror_repeat_sampler);
					cmd.set_specialization_constant(0, true);
				}
				else
				{
					cmd.set_texture(0, 0, *component_ll_views[c][output_level - 1], *mirror_repeat_sampler);
				}

				cmd.set_storage_texture(0, 1, *component_layer_views[c][output_level]);

				char label[64];
				snprintf(label, sizeof(label), "DWT level %u, component %u", output_level, c);
				cmd.begin_region(label);
				{
					cmd.dispatch((push.aligned_resolution.x + 31) / 32, (push.aligned_resolution.y + 31) / 32, 1);
				}
				cmd.end_region();
			}
		}

		cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

		cmd.set_specialization_constant(0, false);
	}

	auto end_dwt = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	device->register_time_interval("GPU", std::move(start_dwt), std::move(end_dwt), "DWT");
	cmd.set_specialization_constant_mask(0);
	return true;
}

size_t Encoder::Impl::compute_num_packets(const void *meta_, size_t packet_boundary) const
{
	auto *meta = static_cast<const BitstreamPacket *>(meta_);
	size_t num_packets = 0;
	size_t size_in_packet = 0;

	size_in_packet += sizeof(BitstreamSequenceHeader);

	for (int i = 0; i < block_count_64x64; i++)
	{
		size_t packet_size = meta[i].num_words * sizeof(uint32_t);
		if (!packet_size)
			continue;

		if (size_in_packet + packet_size > packet_boundary)
		{
			size_in_packet = 0;
			num_packets++;
		}

		size_in_packet += packet_size;
	}

	if (size_in_packet)
		num_packets++;

	return num_packets;
}

void Encoder::Impl::analyze_alternative_packing(const void *mapped_meta, const void *mapped_bitstream) const
{
	auto *meta = static_cast<const BitstreamPacket *>(mapped_meta);
	auto *bitstream = static_cast<const uint32_t *>(mapped_bitstream);

	int num_active_64x64_blocks = 0;
	int num_active_16x16_blocks = 0;
	int small_block_cost = 0;

	for (int component = 0; component < NumComponents; component++)
	{
		for (int level = 0; level < DecompositionLevels; level++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			auto band_width = wavelet_img->get_width(level);
			auto band_height = wavelet_img->get_height(level);
			int blocks_x_64x64 = (band_width + 63) / 64;
			int blocks_y_64x64 = (band_height + 63) / 64;

			for (int band = 3; band >= (level == DecompositionLevels - 1 ? 0 : 1); band--)
			{
				auto &block_mapping = block_meta[component][level][band];

				for (int block_y = 0; block_y < blocks_y_64x64; block_y++)
				{
					for (int block_x = 0; block_x < blocks_x_64x64; block_x++)
					{
						int block_index = block_mapping.block_offset_64x64 + block_y * block_mapping.block_stride_64x64 + block_x;
						if (meta[block_index].num_words == 0)
							continue;

						const auto &mapping = block_64x64_to_16x16_mapping[block_index];

						auto *header = reinterpret_cast<const BitstreamHeader *>(bitstream + meta[block_index].offset_u32);
						int blocks_16x16 = int(Util::popcount32(header->ballot));

						auto *control_words = bitstream + meta[block_index].offset_u32 + 2;
						auto *payload_words = control_words + blocks_16x16;

						Util::for_each_bit(header->ballot, [&](unsigned bit) {
							int x = int(bit & 3);
							int y = int(bit >> 2);
							int block_16x16 = mapping.block_offset_16x16 + mapping.block_stride_16x16 * y + x;
							auto &mapping_16x16 = block_meta_16x16[block_16x16];
							auto q_bits = (*control_words >> 16) & 0xf;

							Util::for_each_bit(mapping_16x16.block_mask, [&](unsigned bit_offset) {
								auto num_planes = q_bits + ((*control_words >> bit_offset) & 0x3);
								if (num_planes != 0)
									num_planes++;

								uint32_t significant_plane[4];
								for (auto &p : significant_plane)
									p = UINT32_MAX;

								for (uint32_t j = 1; j < num_planes - q_bits; j++)
								{
									for (uint32_t k = 0; k < 4; k++)
									{
										bool significant = ((payload_words[j] >> (8 * k)) & 0xff) != 0;
										if (significant)
											significant_plane[k] = std::min<uint32_t>(j - 1, significant_plane[j]);
									}
								}

								const auto cost = [&](uint32_t s) {
									if (s == UINT32_MAX)
										return q_bits ? q_bits + 1 : 0;
									return num_planes - uint32_t(s);
								};

								unsigned repacked_cost = 0;
								for (uint32_t k = 0; k < 4; k++)
									repacked_cost += cost(significant_plane[k]);

								assert(repacked_cost <= num_planes * 4);

								small_block_cost += repacked_cost;
								payload_words += num_planes;
							});

							control_words++;
							num_active_16x16_blocks++;
						});

						auto payload_offset = payload_words - (bitstream + meta[block_index].offset_u32 + meta[block_index].num_words);
						if (payload_offset != 0)
							abort();

						num_active_64x64_blocks++;
					}
				}
			}
		}
	}

	LOGI("64x64 blocks cost: %d (byte cost %zu)\n", num_active_64x64_blocks, num_active_64x64_blocks * 4 * sizeof(BitstreamHeader));
	LOGI("Num active 16x16 blocks %d (byte cost %u):\n", num_active_16x16_blocks, num_active_16x16_blocks * 16);
	LOGI("Small block cost: %d\n", small_block_cost);
}

void Encoder::Impl::report_stats(const void *mapped_meta, const void *mapped_bitstream) const
{
	auto *meta = static_cast<const BitstreamPacket *>(mapped_meta);
	auto *bitstream = static_cast<const uint32_t *>(mapped_bitstream);

	int total_pixels = 0;
	int total_words = 0;

	static const char *components[] = { "Y", "Cb", "Cr" };
	static const char *bands[] = { "LL", "HL", "LH", "HH" };

	constexpr int MaxPlanes = 16;
	int plane_histogram[MaxPlanes][256] = {};
	int total_planes[MaxPlanes] = {};

	for (int component = 0; component < NumComponents; component++)
	{
		for (int level = 0; level < DecompositionLevels; level++)
		{
			int total_words_in_level = 0;

			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			auto band_width = wavelet_img->get_width(level);
			auto band_height = wavelet_img->get_height(level);
			int blocks_x_64x64 = (band_width + 63) / 64;
			int blocks_y_64x64 = (band_height + 63) / 64;

			for (int band = 3; band >= (level == DecompositionLevels - 1 ? 0 : 1); band--)
			{
				auto &block_mapping = block_meta[component][level][band];

				int words = 0;
				for (int block_y = 0; block_y < blocks_y_64x64; block_y++)
				{
					for (int block_x = 0; block_x < blocks_x_64x64; block_x++)
					{
						int block_index = block_mapping.block_offset_64x64 + block_y * block_mapping.block_stride_64x64 + block_x;
						if (meta[block_index].num_words == 0)
							continue;

						const auto &mapping = block_64x64_to_16x16_mapping[block_index];

						words += meta[block_index].num_words;

						auto *header = reinterpret_cast<const BitstreamHeader *>(bitstream + meta[block_index].offset_u32);
						int blocks_16x16 = int(Util::popcount32(header->ballot));

						auto *control_words = bitstream + meta[block_index].offset_u32 + 2;
						auto *payload_words = control_words + blocks_16x16;

						Util::for_each_bit(header->ballot, [&](unsigned bit) {
							int x = int(bit & 3);
							int y = int(bit >> 2);
							int block_16x16 = mapping.block_offset_16x16 + mapping.block_stride_16x16 * y + x;
							auto &mapping_16x16 = block_meta_16x16[block_16x16];
							auto q_bits = (*control_words >> 16) & 0xf;

							Util::for_each_bit(mapping_16x16.block_mask, [&](unsigned bit_offset) {
								auto num_planes = q_bits + ((*control_words >> bit_offset) & 0x3);
								if (num_planes != 0)
									num_planes++;

								for (uint32_t j = 0; j < num_planes; j++)
								{
									plane_histogram[j][(payload_words[j] >> 0) & 0xff]++;
									plane_histogram[j][(payload_words[j] >> 8) & 0xff]++;
									plane_histogram[j][(payload_words[j] >> 16) & 0xff]++;
									plane_histogram[j][(payload_words[j] >> 24) & 0xff]++;
									total_planes[j] += 4;
								}
								payload_words += num_planes;
							});

							control_words++;
						});

						auto payload_offset = payload_words - (bitstream + meta[block_index].offset_u32 + meta[block_index].num_words);
						if (payload_offset != 0)
							abort();
					}
				}

				int bytes = words * 4;
				double bpp = (bytes * 8.0) / (band_width * band_height);

				LOGI("%s: decomposition level %d, band %s: %.3f bpp\n",
				     components[component], level, bands[band], bpp);

				total_words += words;

				if (component == 0)
					total_pixels += band_width * band_height;

				total_words_in_level += words;
			}

			LOGI("%s: decomposition level %d: %d bytes\n", components[component], level, total_words_in_level * 4);
		}
	}

	double plane_entropy[MaxPlanes] = {};

	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < MaxPlanes; j++)
		{
			if (total_planes[j] && plane_histogram[j][i])
			{
				auto p = double(plane_histogram[j][i]) / double(total_planes[j]);
				plane_entropy[j] -= p * log2(p);
			}
		}
	}

	for (int i = 0; i < MaxPlanes; i++)
	{
		LOGI("    Plane %d entropy: %.3f %%\n", i, 100.0 * plane_entropy[i] / 8.0);
		LOGI("    Plane %d bytes: %d\n", i, total_planes[i]);
	}

	LOGI("Overall: %.3f bpp\n", (total_words * 32.0) / total_pixels);

	analyze_alternative_packing(mapped_meta, mapped_bitstream);
}

bool Encoder::Impl::validate_bitstream(
		const uint32_t *bitstream, const BitstreamPacket *meta, uint32_t block_index) const
{
	if (meta[block_index].num_words == 0)
		return true;

	bitstream += meta[block_index].offset_u32;
	auto *header = reinterpret_cast<const BitstreamHeader *>(bitstream);
	if (header->block_index != block_index)
	{
		LOGI("Mismatch in block index. header: %u, meta: %u\n", header->block_index, block_index);
		return false;
	}

	if (header->payload_words != meta[block_index].num_words)
	{
		LOGI("Mismatch in payload words, header: %u, meta: %u\n", header->payload_words, meta[block_index].num_words);
		return false;
	}

	int blocks_16x16 = int(Util::popcount32(header->ballot));
	auto *control_words = bitstream + 2;
	uint32_t offset = 2 + blocks_16x16;

	if (sizeof(*header) / sizeof(uint32_t) + blocks_16x16 > header->payload_words)
	{
		LOGE("payload_words is not large enough.\n");
		return false;
	}

	const auto &mapping = block_64x64_to_16x16_mapping[header->block_index];

	bool invalid_packet = false;

	Util::for_each_bit(header->ballot, [&](unsigned bit) {
		int x = int(bit & 3);
		int y = int(bit >> 2);

		if (x < mapping.block_width_16x16 && y < mapping.block_height_16x16)
		{
			int block_16x16 = mapping.block_offset_16x16 + mapping.block_stride_16x16 * y + x;

			auto &mapping_16x16 = block_meta_16x16[block_16x16];

			auto q_bits = (*control_words >> 16) & 0xf;
			auto lsbs = *control_words & 0x5555u;
			auto msbs = *control_words & 0xaaaau;

			if ((lsbs & (mapping_16x16.block_mask << 0)) != lsbs)
			{
				LOGE("Invalid LSBs for block_index %u.\n", block_index);
				invalid_packet = true;
			}

			if ((msbs & (mapping_16x16.block_mask << 1)) != msbs)
			{
				LOGE("Invalid MSBs for block_index %u.\n", block_index);
				invalid_packet = true;
			}

			auto msbs_shift = msbs >> 1;
			auto sign_mask = (msbs_shift | lsbs) | (q_bits ? mapping_16x16.block_mask : 0);
			msbs |= msbs_shift;
			auto cost = Util::popcount32(lsbs) +
			            Util::popcount32(msbs) +
			            Util::popcount32(sign_mask) +
			            q_bits * mapping_16x16.in_bounds_subblocks;

			offset += cost;
			control_words++;
		}
		else
		{
			LOGE("block_index %u: 16x16 block is out of bounds. (%d, %d) >= (%d, %d)\n",
			     block_index, x, y, mapping.block_width_16x16, mapping.block_height_16x16);
			invalid_packet = true;
		}
	});

	if (invalid_packet)
		return false;

	if (offset != header->payload_words)
	{
		LOGE("Block index %u, offset %u != %u\n", block_index, offset, header->payload_words);
		return false;
	}

	return true;
}

size_t Encoder::Impl::packetize(Packet *packets, size_t packet_boundary,
                                void *output_bitstream_, size_t size,
                                const void *mapped_meta,
                                const void *mapped_bitstream) const
{
	size_t num_packets = 0;
	size_t size_in_packet = 0;
	size_t packet_offset = 0;
	size_t output_offset = 0;
	auto *meta = static_cast<const BitstreamPacket *>(mapped_meta);
	auto *input_bitstream = static_cast<const uint32_t *>(mapped_bitstream);
	auto *output_bitstream = static_cast<uint8_t *>(output_bitstream_);
	(void)size;

	size_t num_non_zero_blocks = 0;
	for (int i = 0; i < block_count_64x64; i++)
		if (meta[i].num_words != 0)
			num_non_zero_blocks++;

	BitstreamSequenceHeader header = {};
	header.width_minus_1 = width - 1;
	header.height_minus_1 = height - 1;
	header.sequence = reinterpret_cast<const BitstreamHeader *>(input_bitstream + meta[0].offset_u32)->sequence;
	header.extended = 1;
	header.code = BITSTREAM_EXTENDED_CODE_START_OF_FRAME;
	header.total_blocks = num_non_zero_blocks;

	assert(sizeof(header) <= size);
	memcpy(output_bitstream, &header, sizeof(header));
	output_offset += sizeof(header);
	size_in_packet += sizeof(header);

	for (int i = 0; i < block_count_64x64; i++)
		if (!validate_bitstream(input_bitstream, meta, i))
			return false;

	for (int i = 0; i < block_count_64x64; i++)
	{
		size_t packet_size = meta[i].num_words * sizeof(uint32_t);
		if (!packet_size)
			continue;

		if (size_in_packet + packet_size > packet_boundary)
		{
			packets[num_packets++] = { packet_offset, size_in_packet };
			size_in_packet = 0;
			packet_offset = output_offset;
		}

		assert(output_offset + packet_size <= size);
		assert(packet_size >= sizeof(BitstreamHeader) / sizeof(uint32_t));

		uint16_t block = reinterpret_cast<const BitstreamHeader *>(input_bitstream + meta[i].offset_u32)->block_index;
		(void)block;
		assert(block == i);

		memcpy(output_bitstream + output_offset, input_bitstream + meta[i].offset_u32, packet_size);

		output_offset += packet_size;
		size_in_packet += packet_size;
	}

	if (size_in_packet)
		packets[num_packets++] = { packet_offset, size_in_packet };

	return num_packets;
}

bool Encoder::Impl::encode(CommandBuffer &cmd, const ViewBuffers &views, const BitstreamBuffers &buffers)
{
	sequence_count = (sequence_count + 1) & SequenceCountMask;

	cmd.image_barrier(*wavelet_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

	cmd.enable_subgroup_size_control(true);

	cmd.fill_buffer(*payload_data, 0, 0, 2 * sizeof(uint32_t));
	cmd.fill_buffer(*bucket_buffer, 0);
	cmd.fill_buffer(*quant_buffer, 0);

	if (!dwt(cmd, views))
		return false;

	// Don't need to read the payload offset counter until quantizer.
	cmd.barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
	            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

	if (!quant(cmd))
		return false;

	if (!analyze_rdo(cmd))
		return false;

	if (!resolve_rdo(cmd, buffers.target_size))
		return false;

	if (!block_packing(cmd, buffers))
		return false;

	cmd.enable_subgroup_size_control(false);
	return true;
}

Encoder::Encoder()
{
	impl.reset(new Impl);
}

bool Encoder::init(Device *device, int width_, int height_)
{
	auto ops = device->get_device_features().vk11_props.subgroupSupportedOperations;
	constexpr VkSubgroupFeatureFlags required_features =
			VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
			VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
			VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
			VK_SUBGROUP_FEATURE_VOTE_BIT |
			VK_SUBGROUP_FEATURE_QUAD_BIT |
			VK_SUBGROUP_FEATURE_BALLOT_BIT |
			VK_SUBGROUP_FEATURE_CLUSTERED_BIT |
			VK_SUBGROUP_FEATURE_BASIC_BIT;

	if ((ops & required_features) != required_features)
	{
		LOGE("There are missing subgroup features. Device supports #%x, but requires #%x.\n",
		     ops, required_features);
		return false;
	}

	if (!device->get_device_features().vk12_features.subgroupBroadcastDynamicId)
		return false;

	if (!device->get_device_features().vk12_features.shaderFloat16)
		return false;

	// This should cover any HW I care about.
	if (!device->supports_subgroup_size_log2(true, 4, 4) &&
	    !device->supports_subgroup_size_log2(true, 5, 5) &&
	    !device->supports_subgroup_size_log2(true, 6, 6))
		return false;

	return impl->init(device, width_, height_);
}

bool Encoder::encode(CommandBuffer &cmd, const ViewBuffers &views, const BitstreamBuffers &buffers)
{
	return impl->encode(cmd, views, buffers);
}

size_t Encoder::compute_num_packets(const void *meta, size_t packet_boundary) const
{
	return impl->compute_num_packets(meta, packet_boundary);
}

size_t Encoder::packetize(Packet *packets, size_t packet_boundary,
                          void *bitstream, size_t size,
                          const void *mapped_meta, const void *mapped_bitstream) const
{
	return impl->packetize(packets, packet_boundary, bitstream, size, mapped_meta, mapped_bitstream);
}

void Encoder::report_stats(const void *mapped_meta, const void *mapped_bitstream) const
{
	impl->report_stats(mapped_meta, mapped_bitstream);
}

uint64_t Encoder::get_meta_required_size() const
{
	return impl->block_count_64x64 * sizeof(BitstreamPacket);
}

Encoder::~Encoder()
{
}
}
