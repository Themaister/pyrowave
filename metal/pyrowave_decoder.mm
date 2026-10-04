// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Metal backend for the PyroWave decoder. Mirrors the compute path of
// pyrowave_decoder.cpp; the fragment iDWT path is not ported. The device object
// and the wavelet pyramid it shares with the encoder live in
// pyrowave_common.mm.

#include "pyrowave_common.hpp"

#ifdef PYROWAVE_METAL_BENCH_HOOKS
#include "pyrowave_bench.h"
#include <chrono>
#include <cmath>
#include <limits>
#endif

#include <memory>
#include <string.h>
#include <vector>

using namespace PyroWave;

namespace
{
// Push constant layouts. These must match the Registers structs SPIRV-Cross
// emitted into shaders/metal/*.metal. MSL gives int2 an 8 byte alignment, so
// the dequant struct is padded out to 24 bytes even though only 20 are used.
struct DequantPush
{
	int32_t resolution[2];
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t padding;
};
static_assert(sizeof(DequantPush) == 24, "DequantPush layout mismatch.");

struct IdwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
};
static_assert(sizeof(IdwtPush) == 16, "IdwtPush layout mismatch.");

// How many decodes a caller may have in flight before acquiring a slot waits for the
// oldest. Two would do for the intended display loop; four is headroom at ~1 MB per
// slot at 4K.
constexpr size_t UploadSlotCount = 4;

struct UploadSlot
{
	id<MTLBuffer> offsets;
	id<MTLBuffer> payload;
	id<MTLCommandBuffer> consumer;

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	id<MTLCounterSampleBuffer> bench_counters;
	MTLTimestamp bench_cpu_start = 0;
	MTLTimestamp bench_gpu_start = 0;
#endif

	void reclaim()
	{
		if (!consumer)
			return;
		// Returns immediately unless the caller really is UploadSlotCount ahead.
		[consumer waitUntilCompleted];
		consumer = nil;
	}

	// No destructor: ARC releases both buffers, and Metal keeps anything a command
	// buffer still references alive on its own.
};

#ifdef PYROWAVE_METAL_BENCH_HOOKS
using BenchClock = std::chrono::steady_clock;

double bench_elapsed_ms(BenchClock::time_point start, BenchClock::time_point end)
{
	return std::chrono::duration<double, std::milli>(end - start).count();
}

pyrowave_bench_decode_timings empty_bench_timings()
{
	const double unavailable = std::numeric_limits<double>::quiet_NaN();
	return { unavailable, unavailable, unavailable, unavailable, unavailable };
}

id<MTLComputeCommandEncoder> create_profiled_encoder(id<MTLCommandBuffer> command, UploadSlot *slot,
	                                                NSUInteger first_sample)
{
	if (@available(macOS 11.0, iOS 14.0, *))
	{
		auto *descriptor = [MTLComputePassDescriptor computePassDescriptor];
		descriptor.dispatchType = MTLDispatchTypeConcurrent;
		auto *attachment = descriptor.sampleBufferAttachments[0];
		attachment.sampleBuffer = slot->bench_counters;
		attachment.startOfEncoderSampleIndex = first_sample;
		attachment.endOfEncoderSampleIndex = first_sample + 1;
		return [command computeCommandEncoderWithDescriptor:descriptor];
	}
	return nil;
}
#endif
}

struct pyrowave_decoder_opaque
{
	pyrowave_device device = nullptr;

	BlockLayout layout;
	BitstreamParser parser;
	WaveletPyramid wavelet;

	// Fixed size on purpose: a vector that appended whenever no slot was free grew
	// without bound when a caller submitted faster than the GPU drained.
	UploadSlot upload_slots[UploadSlotCount];
	size_t next_upload_slot = 0;

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	bool bench_batched_dequant = false;
	bool bench_reduced_idwt_barriers = false;
	int bench_native_dequant = 0;
	bool bench_native_idwt = false;
	bool bench_fused_idwt = false;
	bool bench_compact_fused_idwt = false;
	id<MTLTexture> bench_rgb_output;
	bool bench_profiling = false;
	UploadSlot *bench_last_profile_slot = nullptr;
	pyrowave_bench_decode_timings bench_timings = empty_bench_timings();
#endif
};

namespace
{
// Overallocates so that a steadily sized stream stops reallocating.
bool ensure_buffer(pyrowave_device device, __strong id<MTLBuffer> *buffer, size_t size)
{
	if (*buffer && (*buffer).length >= size)
		return true;

	size_t allocate = size * 2;
	if (allocate < 64 * 1024)
		allocate = 64 * 1024;

	*buffer = [device->mtl newBufferWithLength:allocate options:MTLResourceStorageModeShared];
	if (!*buffer)
	{
		device->log("Failed to allocate a %zu byte upload buffer.", allocate);
		return false;
	}

	return true;
}

UploadSlot *acquire_upload_slot(pyrowave_decoder decoder, size_t offsets_size, size_t payload_size)
{
	UploadSlot *slot = &decoder->upload_slots[decoder->next_upload_slot];
	decoder->next_upload_slot = (decoder->next_upload_slot + 1) % UploadSlotCount;

	// Applies back pressure rather than allocating another slot.
	slot->reclaim();

	if (!ensure_buffer(decoder->device, &slot->offsets, offsets_size) ||
	    !ensure_buffer(decoder->device, &slot->payload, payload_size))
		return nullptr;

	return slot;
}

void encode_dequant(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc, UploadSlot *slot)
{
	auto &layout = decoder->layout;

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	[enc setComputePipelineState:decoder->bench_native_dequant ?
	                           decoder->device->bench_native_dequant_pipeline[decoder->bench_native_dequant - 1] : decoder->device->dequant_pipeline];
#else
	[enc setComputePipelineState:decoder->device->dequant_pipeline];
#endif
	// The u8/u16/u32 aliases of the payload collapse into a single binding in MSL.
	[enc setBuffer:slot->payload offset:0 atIndex:0];
	[enc setBuffer:slot->offsets offset:0 atIndex:2];

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			[enc setTexture:decoder->wavelet.component_layer_views[component][level] atIndex:0];

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				DequantPush push = {};
				push.resolution[0] = layout.level_width(level);
				push.resolution[1] = layout.level_height(level);
				push.output_layer = band;
				push.block_offset_32x32 = layout.block_meta[component][level][band].block_offset_32x32;
				push.block_stride_32x32 = layout.block_meta[component][level][band].block_stride_32x32;
				[enc setBytes:&push length:sizeof(push) atIndex:1];

				[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 31) / 32,
				                                     (push.resolution[1] + 31) / 32, 1)
				      threadsPerThreadgroup:MTLSizeMake(DequantThreadgroupSize, 1, 1)];
			}
		}
	}
}

#ifdef PYROWAVE_METAL_BENCH_HOOKS
void encode_dequant_batched(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc, UploadSlot *slot)
{
	const auto &layout = decoder->layout;
	[enc setComputePipelineState:decoder->bench_native_dequant ?
	                           decoder->device->bench_native_batched_dequant_pipeline[decoder->bench_native_dequant - 1] : decoder->device->bench_batched_dequant_pipeline];
	[enc setBuffer:slot->payload offset:0 atIndex:0];
	[enc setBuffer:slot->offsets offset:0 atIndex:2];

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;
			DequantPush bands[4] = {};
			const int first_band = level == DecompositionLevels - 1 ? 0 : 1;
			const int band_count = 4 - first_band;
			for (int i = 0; i < band_count; i++)
			{
				auto &push = bands[i];
				const int band = first_band + i;
				push.resolution[0] = layout.level_width(level);
				push.resolution[1] = layout.level_height(level);
				push.output_layer = band;
				push.block_offset_32x32 = layout.block_meta[component][level][band].block_offset_32x32;
				push.block_stride_32x32 = layout.block_meta[component][level][band].block_stride_32x32;
			}
			[enc setTexture:decoder->wavelet.component_layer_views[component][level] atIndex:0];
			[enc setBytes:bands length:sizeof(bands) atIndex:1];
			[enc dispatchThreadgroups:MTLSizeMake((layout.level_width(level) + 31) / 32,
			                                     (layout.level_height(level) + 31) / 32, band_count)
			      threadsPerThreadgroup:MTLSizeMake(DequantThreadgroupSize, 1, 1)];
		}
	}
}
#endif

void encode_idwt_dispatch(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc,
                          const IdwtPush &push, id<MTLTexture> input, id<MTLTexture> output,
                          bool dc_shift)
{
#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (decoder->bench_native_idwt)
		[enc setComputePipelineState:decoder->device->bench_native_idwt_pipeline[dc_shift ? 1 : 0]];
	else if (decoder->bench_reduced_idwt_barriers)
		[enc setComputePipelineState:decoder->device->bench_reduced_barrier_idwt_pipeline[dc_shift ? 1 : 0]];
	else
#endif
		[enc setComputePipelineState:decoder->device->idwt_pipeline[dc_shift ? 1 : 0]];
	[enc setBytes:&push length:sizeof(push) atIndex:0];
	[enc setTexture:input atIndex:0];
	[enc setTexture:output atIndex:1];
	[enc setSamplerState:decoder->device->mirror_repeat_sampler atIndex:0];
	[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 15) / 16,
	                                     (push.resolution[1] + 15) / 16, 1)
	      threadsPerThreadgroup:MTLSizeMake(IdwtThreadgroupSize, 1, 1)];
}

void encode_idwt(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc,
                 id<MTLTexture> const planes[3])
{
	auto &layout = decoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
#ifdef PYROWAVE_METAL_BENCH_HOOKS
	const bool fused = decoder->bench_fused_idwt && !decoder->bench_rgb_output &&
	                   layout.level_width(0) >= 32 && layout.level_height(0) >= 32;
#endif

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		// Levels are a dependent chain: this one reads the LL band the previous
		// one produced. Within a level the three components are independent, so
		// the encoder runs concurrently and only the level boundaries barrier.
		if (input_level != DecompositionLevels - 1)
			[enc memoryBarrierWithScope:MTLBarrierScopeTextures];

		IdwtPush push = {};
		// The shader transposes on load, so resolution is swapped here.
		push.resolution[0] = layout.level_height(input_level);
		push.resolution[1] = layout.level_width(input_level);
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

		if (input_level == 0)
		{
#ifdef PYROWAVE_METAL_BENCH_HOOKS
			if (decoder->bench_rgb_output)
			{
				[enc setComputePipelineState:decoder->device->bench_rgb_idwt_pipeline[chroma_420]];
				[enc setBytes:&push length:sizeof(push) atIndex:0];
				for (int c = 0; c < NumComponents; c++)
					[enc setTexture:chroma_420 && c != 0 ? planes[c] : decoder->wavelet.component_layer_views[c][0] atIndex:c];
				[enc setTexture:decoder->bench_rgb_output atIndex:3];
				[enc setSamplerState:decoder->device->mirror_repeat_sampler atIndex:0];
				[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 15) / 16, (push.resolution[1] + 15) / 16, 1)
				      threadsPerThreadgroup:MTLSizeMake(IdwtThreadgroupSize, 1, 1)];
				continue;
			}
#endif
			// Final level writes the output planes directly. Under 420 the chroma
			// planes were already finished one level earlier.
			const int components = chroma_420 ? 1 : NumComponents;
			for (int c = 0; c < components; c++)
			{
#ifdef PYROWAVE_METAL_BENCH_HOOKS
				if (fused)
				{
					struct { IdwtPush fine; IdwtPush coarse; } constants = {};
					constants.fine = push;
					constants.coarse.resolution[0] = layout.level_height(1);
					constants.coarse.resolution[1] = layout.level_width(1);
					constants.coarse.inv_resolution[0] = 1.0f / float(constants.coarse.resolution[0]);
					constants.coarse.inv_resolution[1] = 1.0f / float(constants.coarse.resolution[1]);
					[enc setComputePipelineState:decoder->device->bench_fused_idwt_pipeline[decoder->bench_compact_fused_idwt]];
					[enc setBytes:&constants length:sizeof(constants) atIndex:0];
					[enc setTexture:decoder->wavelet.component_layer_views[c][1] atIndex:0];
					[enc setTexture:decoder->wavelet.component_layer_views[c][0] atIndex:1];
					[enc setTexture:planes[c] atIndex:2];
					[enc setSamplerState:decoder->device->mirror_repeat_sampler atIndex:0];
					[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 15) / 16, (push.resolution[1] + 15) / 16, 1)
					      threadsPerThreadgroup:MTLSizeMake(IdwtThreadgroupSize, 1, 1)];
					continue;
				}
#endif
				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->wavelet.component_layer_views[c][input_level],
				                     planes[c], true);
			}
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
#ifdef PYROWAVE_METAL_BENCH_HOOKS
				if (fused && input_level == 1 && (!chroma_420 || c == 0)) continue;
#endif
				const bool final_chroma = chroma_420 && c != 0 && input_level == 1;
				id<MTLTexture> output = final_chroma ?
				                       planes[c] :
				                       decoder->wavelet.component_ll_views[c][input_level - 1];

				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->wavelet.component_layer_views[c][input_level],
				                     output, final_chroma);
			}
		}
	}
}

bool validate_plane(pyrowave_device device, id<MTLTexture> texture, int index, int width, int height)
{
	if (!texture)
	{
		device->log("Output plane %d is NULL.", index);
		return false;
	}

	if (texture.textureType != MTLTextureType2D)
	{
		device->log("Output plane %d must be MTLTextureType2D.", index);
		return false;
	}

	if (int(texture.width) != width || int(texture.height) != height)
	{
		device->log("Output plane %d is %ux%u, expected %dx%d.",
		            index, unsigned(texture.width), unsigned(texture.height), width, height);
		return false;
	}

	if ((texture.usage & MTLTextureUsageShaderWrite) == 0)
	{
		device->log("Output plane %d was not created with MTLTextureUsageShaderWrite.", index);
		return false;
	}

	return true;
}
}

//////
// Public API

pyrowave_result pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder)
{
	if (!info || !decoder || !info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma != PYROWAVE_CHROMA_SUBSAMPLING_420 &&
	    info->chroma != PYROWAVE_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	const bool chroma_420 = info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	if (chroma_420 && ((info->width & 1) != 0 || (info->height & 1) != 0))
	{
		info->device->log("420 subsampling requires even dimensions, got %dx%d.",
		                  info->width, info->height);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	auto created = std::unique_ptr<pyrowave_decoder_opaque>(new (std::nothrow) pyrowave_decoder_opaque);
	if (!created)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;

	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	created->parser.init(&created->layout);

	if (!created->wavelet.init(created->device, created->layout))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	*decoder = created.release();
	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_destroy(pyrowave_decoder decoder)
{
	delete decoder;
}

void pyrowave_decoder_clear(pyrowave_decoder decoder)
{
	if (decoder)
		decoder->parser.clear();
}

pyrowave_result pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size)
{
	if (!decoder || (!data && size != 0))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (!decoder->parser.push_packet(data, size))
		return PYROWAVE_ERROR_CORRUPT_BITSTREAM;

	return PYROWAVE_SUCCESS;
}

bool pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame)
{
	return decoder && decoder->parser.decode_is_ready(allow_partial_frame);
}

bool pyrowave_decoder_decode_is_ready_with_sideband(pyrowave_decoder decoder, bool allow_partial_frame,
                                                    int num_pristine_bands, float minimum_packet_ratio,
                                                    const uint32_t *active_block_mask, size_t word_count)
{
	// num_pristine_bands is range checked only by an assert in has_pristine_bands(),
	// matching the Vulkan C API. It indexes block_meta[..][DecompositionLevels - band][..],
	// so a large enough value walks off the front of the array with NDEBUG.
	if (!decoder)
		return false;

	return decoder->parser.decode_is_ready(allow_partial_frame, num_pristine_bands, minimum_packet_ratio,
	                                       active_block_mask, word_count);
}

pyrowave_result pyrowave_decoder_decode_gpu_buffer(pyrowave_decoder decoder,
                                                   pyrowave_mtl_command_buffer command_buffer,
                                                   const pyrowave_gpu_buffers *buffers)
{
	if (!decoder || !command_buffer || !buffers)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto &layout = decoder->layout;
	auto *device = decoder->device;

	id<MTLTexture> planes[3];
	const int chroma_width = layout.chroma == ChromaSubsampling::Chroma420 ?
	                         layout.width / 2 : layout.width;
	const int chroma_height = layout.chroma == ChromaSubsampling::Chroma420 ?
	                          layout.height / 2 : layout.height;

	for (int i = 0; i < 3; i++)
	{
		planes[i] = (__bridge id<MTLTexture>)(buffers->planes[i]);
		const int expected_width = i == 0 ? layout.width : chroma_width;
		const int expected_height = i == 0 ? layout.height : chroma_height;
		if (!validate_plane(device, planes[i], i, expected_width, expected_height))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	const auto &offsets = decoder->parser.dequant_offsets();
	const auto &payload = decoder->parser.payload();

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	const bool profiling = decoder->bench_profiling;
	BenchClock::time_point profile_start;
	if (profiling)
	{
		decoder->bench_last_profile_slot = nullptr;
		decoder->bench_timings = empty_bench_timings();
		profile_start = BenchClock::now();
	}
#endif

	const size_t offsets_size = offsets.size() * sizeof(uint32_t);
	// The dequant shader can read slightly past the end of the payload, so pad.
	const size_t payload_size = payload.size() * sizeof(uint32_t) + 16;

	auto *slot = acquire_upload_slot(decoder, offsets_size, payload_size);
	if (!slot)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	if (offsets_size)
		memcpy(slot->offsets.contents, offsets.data(), offsets_size);
	if (!payload.empty())
		memcpy(slot->payload.contents, payload.data(), payload.size() * sizeof(uint32_t));

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (profiling)
	{
		decoder->bench_timings.upload_cpu_ms = bench_elapsed_ms(profile_start, BenchClock::now());
		// GPU counter clocks may differ from CPU clocks. Bracket sampling with
		// paired references, then calibrate durations after command completion.
		[device->mtl sampleTimestamps:&slot->bench_cpu_start gpuTimestamp:&slot->bench_gpu_start];
		profile_start = BenchClock::now();
	}
#endif

	auto *cmd = (__bridge id<MTLCommandBuffer>)(command_buffer);

	// Every dequant dispatch writes a distinct (component, level, band) region of
	// the pyramid and none reads another's output, so they can all run at once. A
	// serial encoder would barrier between all ~42, but the cost is underutilization
	// rather than barrier latency: each dispatch is far too small to fill the GPU on
	// its own, which is why this pays at low resolution and not at 1080p 4:4:4.
	id<MTLComputeCommandEncoder> dequant_enc;
#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (profiling)
		dequant_enc = create_profiled_encoder(cmd, slot, 0);
	else
#endif
		dequant_enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	if (!dequant_enc)
		return PYROWAVE_ERROR_GENERIC;

	dequant_enc.label = @("pyrowave dequant");
#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (decoder->bench_batched_dequant)
		encode_dequant_batched(decoder, dequant_enc, slot);
	else
#endif
		encode_dequant(decoder, dequant_enc, slot);
	[dequant_enc endEncoding];

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (profiling)
	{
		decoder->bench_timings.dequant_encode_cpu_ms = bench_elapsed_ms(profile_start, BenchClock::now());
		profile_start = BenchClock::now();
	}
#endif

	// The iDWT is a dependent chain across levels, but the three components within
	// a level are independent, so this is also concurrent with explicit barriers
	// at the level boundaries only. Ordering against the dequant work above comes
	// from the encoder boundary, which Metal tracks automatically.
	id<MTLComputeCommandEncoder> idwt_enc;
#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (profiling)
		idwt_enc = create_profiled_encoder(cmd, slot, 2);
	else
#endif
		idwt_enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	if (!idwt_enc)
		return PYROWAVE_ERROR_GENERIC;

	idwt_enc.label = @("pyrowave idwt");
	encode_idwt(decoder, idwt_enc, planes);
	[idwt_enc endEncoding];

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (profiling)
		decoder->bench_timings.idwt_encode_cpu_ms = bench_elapsed_ms(profile_start, BenchClock::now());
#endif

	// Retained until this slot comes round again, so its buffers cannot be rewritten
	// while the GPU is still reading them.
	slot->consumer = cmd;

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	if (profiling)
		decoder->bench_last_profile_slot = slot;
#endif

	decoder->parser.mark_frame_decoded();
	return PYROWAVE_SUCCESS;
}

#ifdef PYROWAVE_METAL_BENCH_HOOKS
extern "C" pyrowave_result pyrowave_bench_set_native_dequant(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder) return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled) { auto result = ensure_native_dequant_pipelines(decoder->device); if (result != PYROWAVE_SUCCESS) return result; }
	decoder->bench_native_dequant = enabled;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_hybrid_dequant(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder) return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled) { auto result = ensure_native_dequant_pipelines(decoder->device, true); if (result != PYROWAVE_SUCCESS) return result; }
	decoder->bench_native_dequant = enabled ? 2 : 0;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_native_idwt(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder) return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled) { auto result = ensure_native_idwt_pipelines(decoder->device); if (result != PYROWAVE_SUCCESS) return result; }
	decoder->bench_native_idwt = enabled;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_fused_idwt(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder) return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled) { auto result = ensure_fused_idwt_pipeline(decoder->device); if (result != PYROWAVE_SUCCESS) return result; }
	decoder->bench_fused_idwt = enabled;
	decoder->bench_compact_fused_idwt = false;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_compact_fused_idwt(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder) return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled) { auto result = ensure_fused_idwt_pipeline(decoder->device, true); if (result != PYROWAVE_SUCCESS) return result; }
	decoder->bench_fused_idwt = enabled;
	decoder->bench_compact_fused_idwt = enabled;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_rgb_output(pyrowave_decoder decoder, pyrowave_mtl_texture texture)
{
	if (!decoder)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	auto output = (__bridge id<MTLTexture>)texture;
	if (texture)
	{
		if (output.width != NSUInteger(decoder->layout.width) || output.height != NSUInteger(decoder->layout.height) ||
		    output.pixelFormat != MTLPixelFormatRGBA8Unorm || !(output.usage & MTLTextureUsageShaderWrite))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		auto result = ensure_rgb_idwt_pipeline(decoder->device, decoder->layout.chroma == ChromaSubsampling::Chroma420);
		if (result != PYROWAVE_SUCCESS) return result;
	}
	decoder->bench_rgb_output = output;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_batched_dequant(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled)
	{
		const auto result = ensure_batched_dequant_pipeline(decoder->device);
		if (result != PYROWAVE_SUCCESS)
			return result;
	}
	decoder->bench_batched_dequant = enabled;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_reduced_idwt_barriers(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (enabled)
	{
		const auto result = ensure_reduced_barrier_idwt_pipelines(decoder->device);
		if (result != PYROWAVE_SUCCESS)
			return result;
	}
	decoder->bench_reduced_idwt_barriers = enabled;
	return PYROWAVE_SUCCESS;
}

extern "C" pyrowave_result pyrowave_bench_set_decode_profiling(pyrowave_decoder decoder, bool enabled)
{
	if (!decoder)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	decoder->bench_profiling = false;
	decoder->bench_last_profile_slot = nullptr;
	decoder->bench_timings = empty_bench_timings();
	if (!enabled)
		return PYROWAVE_SUCCESS;

	if (@available(macOS 11.0, iOS 14.0, *))
	{
		auto *device = decoder->device;
		if (![device->mtl supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
		{
			device->log("GPU stage timestamp sampling is unavailable on this device.");
			return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
		}
		id<MTLCounterSet> timestamps = nil;
		for (id<MTLCounterSet> candidate in device->mtl.counterSets)
		{
			if ([candidate.name isEqualToString:MTLCommonCounterSetTimestamp])
			{
				timestamps = candidate;
				break;
			}
		}
		if (!timestamps)
		{
			device->log("GPU timestamp counter set is unavailable on this device.");
			return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
		}
		auto *descriptor = [MTLCounterSampleBufferDescriptor new];
		descriptor.counterSet = timestamps;
		descriptor.sampleCount = 4;
		descriptor.storageMode = MTLStorageModeShared;
		descriptor.label = @"pyrowave decode stage timestamps";
		for (auto &slot : decoder->upload_slots)
		{
			if (slot.bench_counters)
				continue;
			NSError *error = nil;
			slot.bench_counters = [device->mtl newCounterSampleBufferWithDescriptor:descriptor error:&error];
			if (!slot.bench_counters)
			{
				device->log("Failed to create stage timestamp buffer: %s",
				            error.localizedDescription.UTF8String ?: "unknown error");
				return PYROWAVE_ERROR_GENERIC;
			}
		}
		decoder->bench_profiling = true;
		return PYROWAVE_SUCCESS;
	}
	decoder->device->log("GPU stage profiling requires macOS 11 or iOS 14.");
	return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
}

extern "C" pyrowave_result pyrowave_bench_get_decode_timings(pyrowave_decoder decoder,
	                                                        pyrowave_bench_decode_timings *timings)
{
	if (!decoder || !timings)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	*timings = decoder->bench_timings;
	auto *slot = decoder->bench_last_profile_slot;
	if (!decoder->bench_profiling || !slot || slot->consumer.status != MTLCommandBufferStatusCompleted)
		return PYROWAVE_ERROR_GENERIC;

	if (@available(macOS 11.0, iOS 14.0, *))
	{
		MTLTimestamp cpu_end = 0, gpu_end = 0;
		[decoder->device->mtl sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];
		NSData *resolved = [slot->bench_counters resolveCounterRange:NSMakeRange(0, 4)];
		if (resolved.length < 4 * sizeof(MTLCounterResultTimestamp) ||
		    cpu_end <= slot->bench_cpu_start || gpu_end <= slot->bench_gpu_start)
			return PYROWAVE_SUCCESS; // GPU fields remain NaN rather than reporting a false zero.
		const auto *samples = static_cast<const MTLCounterResultTimestamp *>(resolved.bytes);
		// sampleTimestamps CPU values are nanoseconds, per Apple's counter conversion API.
		const double milliseconds_per_gpu_tick = double(cpu_end - slot->bench_cpu_start) /
		                                         double(gpu_end - slot->bench_gpu_start) / 1e6;
		auto duration = [&](NSUInteger first) {
			const uint64_t start = samples[first].timestamp, end = samples[first + 1].timestamp;
			if (start == MTLCounterErrorValue || end == MTLCounterErrorValue || end <= start ||
			    start < slot->bench_gpu_start || end > gpu_end)
				return std::numeric_limits<double>::quiet_NaN();
			const double elapsed = double(end - start) * milliseconds_per_gpu_tick;
			return std::isfinite(elapsed) && elapsed > 0.0 ? elapsed : std::numeric_limits<double>::quiet_NaN();
		};
		timings->dequant_gpu_ms = duration(0);
		timings->idwt_gpu_ms = duration(2);
		return PYROWAVE_SUCCESS;
	}
	return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
}
#endif
