// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Metal backend for the PyroWave encoder. Mirrors the compute pipeline of
// pyrowave_encoder.cpp: dwt -> quant -> analyze_rdo -> resolve_rdo ->
// block_packing on the GPU, then packetize on the CPU.
//
// Two things differ structurally from the Vulkan encoder. It owns its command queue
// rather than taking a caller command buffer, because the packet queries have to block
// on completion. And the result buffers are shared storage read directly by the CPU;
// the Vulkan side stages them through device local memory because host visible writes
// are slow on discrete GPUs, which does not apply to unified memory.

#include "pyrowave_common.hpp"

#include "shaders/pyrowave_msl.h"

#import <IOSurface/IOSurfaceRef.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdlib.h>
#include <string.h>

using namespace PyroWave;

namespace
{
// Push constant layouts, matching the Registers structs SPIRV-Cross emitted into
// shaders/metal/*.metal. MSL aligns int2 and float2 to 8 bytes, so several need
// explicit tail padding.
struct DwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
	int32_t aligned_resolution[2];
};
static_assert(sizeof(DwtPush) == 24, "DwtPush layout mismatch.");

struct QuantPush
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	float inv_resolution[2];
	float input_layer;
	float quant_resolution;
	int32_t block_offset;
	int32_t block_stride;
	float rdo_distortion_scale;
	float padding;
};
static_assert(sizeof(QuantPush) == 48, "QuantPush layout mismatch.");

struct AnalyzePush
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	uint32_t total_wg_count;
	uint32_t num_blocks_aligned;
	uint32_t block_index_shamt;
	uint32_t padding;
};
static_assert(sizeof(AnalyzePush) == 48, "AnalyzePush layout mismatch.");

struct ResolvePush
{
	uint32_t target_payload_size;
	uint32_t num_blocks_per_subdivision;
};
static_assert(sizeof(ResolvePush) == 8, "ResolvePush layout mismatch.");

struct BlockPackingPush
{
	int32_t resolution[2];
	int32_t resolution_32x32_blocks[2];
	int32_t resolution_8x8_blocks[2];
	uint32_t quant_resolution_code;
	uint32_t sequence_code;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
};
static_assert(sizeof(BlockPackingPush) == 48, "BlockPackingPush layout mismatch.");

//////
// Initial quantization resolution. Pure float math lifted from
// pyrowave_encoder.cpp; the only change is that the precision and chroma mode are
// passed in rather than read off Configuration and WaveletBuffers.

float get_noise_power_normalized_quant_resolution(int level, int component, int band, int precision)
{
	// The initial quantization resolution aims for a flat spectrum with noise power
	// normalization. The low-pass gain for CDF 9/7 is 6 dB (1 bit). Every
	// decomposition level subtracts 6 dB.
	int bits = precision >= 1 ? 8 : 6;

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

float get_quant_resolution(int level, int component, int band, int precision)
{
	// FP16 range is limited, and this is more than a good enough initial estimate.
	return std::min<float>(precision >= 1 ? 4096.0f : 512.0f,
	                       get_noise_power_normalized_quant_resolution(level, component, band, precision));
}

float get_quant_rdo_distortion_scale(int level, int component, int band, int precision,
                                     ChromaSubsampling chroma)
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
	{
		// Consider chroma a little more important if we're not subsampling.
		if (chroma == ChromaSubsampling::Chroma420)
			csf *= 0.6f;
	}

	// Due to filtering, distortion in lower bands will result in more noise power.
	// By scaling the distortion by this factor, we ensure uniform results.
	float resolution = get_noise_power_normalized_quant_resolution(level, component, band, precision);
	float weighted_resolution = csf * resolution;

	// The distortion is scaled in terms of power, not amplitude.
	return weighted_resolution * weighted_resolution;
}

#ifdef PYROWAVE_METAL_BENCH_HOOKS
// Lets tools/metal/bench_encode.mm measure the concurrent encoder against a serial one.
// Re-read per call rather than cached, so the tool can interleave both modes within one
// run -- background GPU load drifts enough between runs to fake a result otherwise.
bool bench_serial_dispatch()
{
	const char *env = getenv("PYROWAVE_BENCH_SERIAL");
	return env && env[0] == '1';
}
#else
constexpr bool bench_serial_dispatch()
{
	return false;
}
#endif

// The explicit barriers are only needed, and only legal, on a concurrent encoder.
// Outside a benchmark build this folds away and the barrier is unconditional.
void stage_barrier(id<MTLComputeCommandEncoder> enc, MTLBarrierScope scope)
{
	if (!bench_serial_dispatch())
		[enc memoryBarrierWithScope:scope];
}

uint32_t floor_log2(uint32_t v)
{
	uint32_t result = 0;
	while (v > 1)
	{
		v >>= 1;
		result++;
	}
	return result;
}

// The three single channel textures the DWT samples at level 0. For NV12 input two are
// swizzled views of one chroma texture, so the backing objects are tracked separately.
struct InputTextures
{
	id<MTLTexture> sampled[NumComponents] = {};
	// Worst case is NV12: the luma plane, the interleaved chroma plane and its two
	// swizzled views. The three plane paths only ever use three.
	static constexpr int MaxOwned = 4;
	id<MTLTexture> owned[MaxOwned] = {};
	int num_owned = 0;

	id<MTLTexture> adopt(id<MTLTexture> texture)
	{
		if (texture)
		{
			assert(num_owned < MaxOwned);
			owned[num_owned++] = texture;
		}
		return texture;
	}

	void release_all()
	{
		// sampled[] aliases owned[] for NV12, but each is an independent strong
		// reference under ARC, so both arrays have to be cleared.
		for (int i = 0; i < num_owned; i++)
			owned[i] = nil;
		num_owned = 0;
		for (int i = 0; i < NumComponents; i++)
			sampled[i] = nil;
	}

	~InputTextures() { release_all(); }
	InputTextures() = default;
	InputTextures(const InputTextures &) = delete;
	void operator=(const InputTextures &) = delete;
};
}

struct pyrowave_encoder_opaque
{
	pyrowave_device device = nullptr;

	BlockLayout layout;
	WaveletPyramid wavelet;

	id<MTLCommandQueue> queue;

	// Device local scratch, sized once from the block layout.
	id<MTLBuffer> bucket_buffer;
	id<MTLBuffer> meta_buffer;
	id<MTLBuffer> block_stat_buffer;
	id<MTLBuffer> payload_data;
	id<MTLBuffer> quant_buffer;

	// Results. Shared storage so the packet queries can read them without a copy.
	id<MTLBuffer> bitstream_meta;
	id<MTLBuffer> bitstream;

	// Input textures for the CPU entry point, reused across frames. The GPU entry
	// point wraps the caller's IOSurfaces per call instead.
	InputTextures cpu_input;
	pyrowave_cpu_buffer_format cpu_input_format = PYROWAVE_CPU_BUFFER_FORMAT_INT_MAX;

	// Retained until the next encode replaces it, so the packet queries have
	// something to block on.
	id<MTLCommandBuffer> pending;
	bool have_result = false;

#ifdef PYROWAVE_METAL_BENCH_HOOKS
	double bench_last_gpu_ms = 0.0;
#endif

	uint32_t sequence_count = 0;

	// No destructor: ARC releases every buffer, the queue and any pending command
	// buffer, and Metal keeps whatever an in-flight encode still references alive.
};

namespace
{
size_t bucket_buffer_size(const BlockLayout &layout)
{
	size_t size = RDOBucketOffset;
	size += size_t(NumRDOBuckets) * BlockSpaceSubdivision * sizeof(uint32_t);
	size += size_t(NumRDOBuckets) * compute_block_count_per_subdivision(layout.block_count_32x32) *
	        BlockSpaceSubdivision * sizeof(RDOperation);
	return size;
}

bool ensure_encode_pipelines(pyrowave_device device)
{
	std::lock_guard<std::mutex> holder{device->encode_pipeline_lock};
	if (device->encode_pipelines_ready)
		return true;

	// Each shader is its own MTLLibrary. They cannot be concatenated: SPIRV-Cross emits
	// its own spv* helpers and a differently shaped Registers into every source.
	const char *dwt_source;
	switch (device->precision)
	{
	case 0: dwt_source = dwt_fp16_msl_source; break;
	case 1: dwt_source = dwt_fp16_storage_msl_source; break;
	default: dwt_source = dwt_msl_source; break;
	}

	char dwt_label[32];
	snprintf(dwt_label, sizeof(dwt_label), "dwt (precision %d)", device->precision);

	if (auto *library = compile_library(device, dwt_source, dwt_label))
	{
		// Indexed by DCShift, which converts the unorm input planes into the signed
		// range the wavelet transform works in.
		for (int i = 0; i < 2; i++)
		{
			device->dwt_pipeline[i] = create_pipeline_bool_constant(
					device, library, "pyrowave_dwt", DwtThreadgroupSize, 0, i != 0);
		}
	}

	// SkipQuantScale is function constant 1 and wants its default of false, but Metal
	// will not build a pipeline from an unspecialized function that declares any
	// constants, so it has to be set explicitly.
	if (auto *library = compile_library(device, wavelet_quant_msl_source, "wavelet_quant"))
	{
		device->quant_pipeline = create_pipeline_bool_constant(
				device, library, "pyrowave_wavelet_quant", QuantThreadgroupSize, 1, false);
	}

	struct
	{
		__strong id<MTLComputePipelineState> *pipeline;
		const char *source;
		const char *entry_point;
		uint32_t threads;
	} const simple[] = {
		{ &device->analyze_pipeline, analyze_rate_control_msl_source,
		  "pyrowave_analyze_rate_control", AnalyzeThreadgroupSize },
		{ &device->analyze_finalize_pipeline, analyze_rate_control_finalize_msl_source,
		  "pyrowave_analyze_rate_control_finalize", AnalyzeFinalizeThreadgroupSize },
		{ &device->block_packing_pipeline, block_packing_msl_source,
		  "pyrowave_block_packing", BlockPackingThreadgroupSize },
	};

	for (auto &shader : simple)
	{
		if (auto *library = compile_library(device, shader.source, shader.entry_point))
		{
			*shader.pipeline = create_pipeline(device, library, shader.entry_point, shader.threads);
		}
	}

	// resolve_rate_control declares its workgroup size as a specialization
	// constant and requires it to equal the SIMD width, so the whole threadgroup
	// is one SIMD group and its subgroup scan needs no cross-group communication.
	if (auto *library = compile_library(device, resolve_rate_control_msl_source, "resolve_rate_control"))
	{
		uint32_t threadgroup_size = ResolveThreadgroupSize;
		auto constants = [MTLFunctionConstantValues new];
		[constants setConstantValue:&threadgroup_size type:MTLDataTypeUInt atIndex:0];
		device->resolve_pipeline = create_pipeline(device, library, "pyrowave_resolve_rate_control",
		                                           ResolveThreadgroupSize, constants);
	}

	id<MTLComputePipelineState> const required[] = {
		device->dwt_pipeline[0], device->dwt_pipeline[1], device->quant_pipeline,
		device->analyze_pipeline, device->analyze_finalize_pipeline,
		device->resolve_pipeline, device->block_packing_pipeline,
	};

	for (auto *pipeline : required)
		if (!pipeline)
			return false;

	if (device->resolve_pipeline.threadExecutionWidth != ResolveThreadgroupSize)
	{
		device->log("resolve_rate_control needs a SIMD width of %u, but the device reports %u.",
		            ResolveThreadgroupSize,
		            unsigned(device->resolve_pipeline.threadExecutionWidth));
		return false;
	}

	// The quantizer samples with a border so coefficients past the edge of a band
	// read as zero, rather than the DWT's mirror repeat.
	auto sampler_desc = [MTLSamplerDescriptor new];
	sampler_desc.minFilter = MTLSamplerMinMagFilterNearest;
	sampler_desc.magFilter = MTLSamplerMinMagFilterNearest;
	sampler_desc.mipFilter = MTLSamplerMipFilterNearest;
	sampler_desc.sAddressMode = MTLSamplerAddressModeClampToBorderColor;
	sampler_desc.tAddressMode = MTLSamplerAddressModeClampToBorderColor;
	sampler_desc.rAddressMode = MTLSamplerAddressModeClampToBorderColor;
	sampler_desc.borderColor = MTLSamplerBorderColorTransparentBlack;
	device->border_sampler = [device->mtl newSamplerStateWithDescriptor:sampler_desc];

	if (!device->border_sampler)
		return false;

	device->encode_pipelines_ready = true;
	return true;
}

id<MTLBuffer> create_scratch_buffer(pyrowave_device device, size_t size, MTLResourceOptions options,
                                   const char *label)
{
	auto buffer = [device->mtl newBufferWithLength:size options:options];
	if (!buffer)
	{
		device->log("Failed to allocate a %zu byte %s buffer.", size, label);
		return nullptr;
	}
	buffer.label = @(label);
	return buffer;
}

bool create_encode_buffers(pyrowave_encoder encoder)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;

	struct
	{
		__strong id<MTLBuffer> *buffer;
		size_t size;
		const char *label;
	} const scratch[] = {
		{ &encoder->block_stat_buffer,
		  size_t(layout.block_count_8x8) * sizeof(BlockStatsBlock), "pyrowave-block-stats" },
		{ &encoder->meta_buffer,
		  size_t(layout.block_count_8x8) * sizeof(BlockMeta), "pyrowave-block-meta" },
		// Worst case estimate, same as the Vulkan encoder's. The first two words are
		// allocation counters and the coefficient payload starts at byte 8.
		{ &encoder->payload_data,
		  size_t(layout.aligned_width) * size_t(layout.aligned_height) * 2, "pyrowave-payload" },
		{ &encoder->quant_buffer,
		  size_t(layout.block_count_32x32) * sizeof(uint32_t), "pyrowave-quant" },
		{ &encoder->bucket_buffer, bucket_buffer_size(layout), "pyrowave-buckets" },
	};

	for (auto &entry : scratch)
	{
		*entry.buffer = create_scratch_buffer(device, entry.size,
		                                      MTLResourceStorageModePrivate, entry.label);
		if (!*entry.buffer)
			return false;
	}

	encoder->bitstream_meta = create_scratch_buffer(
			device, size_t(layout.block_count_32x32) * sizeof(BitstreamPacket),
			MTLResourceStorageModeShared, "pyrowave-bitstream-meta");

	return encoder->bitstream_meta != nullptr;
}

// Sized per encode, since the rate control target can change per frame. Releasing the
// old one mid-flight is safe: Metal keeps it alive while a command buffer references it.
bool ensure_bitstream_buffer(pyrowave_encoder encoder, size_t size)
{
	if (encoder->bitstream && encoder->bitstream.length >= size)
		return true;


	encoder->bitstream = create_scratch_buffer(encoder->device, size,
	                                           MTLResourceStorageModeShared, "pyrowave-bitstream");
	return encoder->bitstream != nullptr;
}

id<MTLTexture> create_input_texture(pyrowave_device device, MTLPixelFormat format,
                                   int width, int height, bool needs_view)
{
	auto desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType2D;
	desc.pixelFormat = format;
	desc.width = width;
	desc.height = height;
	desc.usage = MTLTextureUsageShaderRead |
	               (needs_view ? MTLTextureUsagePixelFormatView : 0);
	// Written by the CPU, and IOSurface backed textures cannot be private anyway.
	desc.storageMode = MTLStorageModeShared;
	desc.mipmapLevelCount = 1;

	auto texture = [device->mtl newTextureWithDescriptor:desc];

	if (!texture)
		device->log("Failed to allocate a %dx%d input texture.", width, height);

	return texture;
}

id<MTLTexture> wrap_surface_plane(pyrowave_device device, IOSurfaceRef surface, int plane,
                                 MTLPixelFormat format, int width, int height, bool needs_view)
{
	auto desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType2D;
	desc.pixelFormat = format;
	desc.width = width;
	desc.height = height;
	desc.usage = MTLTextureUsageShaderRead |
	               (needs_view ? MTLTextureUsagePixelFormatView : 0);
	desc.storageMode = MTLStorageModeShared;
	desc.mipmapLevelCount = 1;

	id<MTLTexture> texture = [device->mtl newTextureWithDescriptor:desc iosurface:surface plane:plane];

	if (!texture)
		device->log("Failed to wrap plane %d of the input IOSurface.", plane);

	return texture;
}

// dwt gathers the red channel of whatever it is handed, so an interleaved chroma plane
// is bound twice with the wanted component broadcast into red -- what the Vulkan API
// documents for NV12 too ("pass in the same plane for Cb and Cr, but use swizzle"). An
// R8 format view of an RG8 texture is impossible, 8 versus 16 bpp, so only the swizzle
// changes.
bool make_interleaved_chroma_views(pyrowave_device device, InputTextures &input, id<MTLTexture> chroma)
{
	static const MTLTextureSwizzle components[2] = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen };

	for (int i = 0; i < 2; i++)
	{
		const auto c = components[i];
		input.sampled[i + 1] = input.adopt(
				[chroma newTextureViewWithPixelFormat:chroma.pixelFormat
				                         textureType:MTLTextureType2D
				                              levels:NSMakeRange(0, 1)
				                              slices:NSMakeRange(0, 1)
				                             swizzle:MTLTextureSwizzleChannelsMake(c, c, c, c)]);

		if (!input.sampled[i + 1])
		{
			device->log("Failed to create a swizzled view of the interleaved chroma plane.");
			return false;
		}
	}

	return true;
}

bool ensure_cpu_input(pyrowave_encoder encoder, pyrowave_cpu_buffer_format format)
{
	if (encoder->cpu_input_format == format && encoder->cpu_input.sampled[0])
		return true;

	encoder->cpu_input.release_all();
	encoder->cpu_input_format = PYROWAVE_CPU_BUFFER_FORMAT_INT_MAX;

	auto *device = encoder->device;
	auto &input = encoder->cpu_input;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int chroma_width = chroma_420 ? layout.width / 2 : layout.width;
	const int chroma_height = chroma_420 ? layout.height / 2 : layout.height;

	input.sampled[0] = input.adopt(create_input_texture(device, MTLPixelFormatR8Unorm,
	                                                    layout.width, layout.height, false));
	if (!input.sampled[0])
		return false;

	if (format == PYROWAVE_CPU_BUFFER_FORMAT_NV12)
	{
		auto *chroma = input.adopt(create_input_texture(device, MTLPixelFormatRG8Unorm,
		                                                chroma_width, chroma_height, true));
		if (!chroma || !make_interleaved_chroma_views(device, input, chroma))
			return false;
	}
	else
	{
		for (int i = 1; i < NumComponents; i++)
		{
			input.sampled[i] = input.adopt(create_input_texture(device, MTLPixelFormatR8Unorm,
			                                                    chroma_width, chroma_height, false));
			if (!input.sampled[i])
				return false;
		}
	}

	encoder->cpu_input_format = format;
	return true;
}

//////
// GPU dispatch, one function per stage of pyrowave_encoder.cpp's pipeline.

void dispatch_dwt(pyrowave_encoder encoder, id<MTLComputeCommandEncoder> enc,
                id<MTLTexture> const planes[NumComponents])
{
	const auto &layout = encoder->layout;
	auto &wavelet = encoder->wavelet;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	[enc setSamplerState:encoder->device->mirror_repeat_sampler atIndex:0];

	for (int output_level = 0; output_level < DecompositionLevels; output_level++)
	{
		// Each level transforms the LL band the previous one produced, so the levels
		// are a dependent chain. The components within a level are independent.
		if (output_level != 0)
			stage_barrier(enc, MTLBarrierScopeTextures);

		DwtPush level_push = {};
		if (output_level == 0)
		{
			// The input planes are only `width` wide, but the transform covers the
			// aligned extent and mirrors the source to fill it.
			level_push.resolution[0] = layout.width;
			level_push.resolution[1] = layout.height;
			level_push.aligned_resolution[0] = layout.aligned_width;
			level_push.aligned_resolution[1] = layout.aligned_height;
		}
		else
		{
			level_push.resolution[0] = layout.level_width(output_level - 1);
			level_push.resolution[1] = layout.level_height(output_level - 1);
			level_push.aligned_resolution[0] = level_push.resolution[0];
			level_push.aligned_resolution[1] = level_push.resolution[1];
		}

		// Under 420, chroma is half resolution and so enters the pyramid at level 1
		// rather than level 0.
		const int components = (output_level == 0 && chroma_420) ? 1 : NumComponents;

		for (int component = 0; component < components; component++)
		{
			DwtPush push = level_push;
			id<MTLTexture> input;
			// DCShift subtracts 0.5 to centre unorm input on zero, so it applies
			// exactly when the source is an input plane rather than a previous level.
			bool reads_input_plane = output_level == 0;

			if (output_level == 0)
			{
				input = planes[component];
			}
			else if (chroma_420 && component != 0 && output_level == 1)
			{
				input = planes[component];
				reads_input_plane = true;
				push.resolution[0] = layout.width / 2;
				push.resolution[1] = layout.height / 2;
				push.aligned_resolution[0] = layout.aligned_width >> output_level;
				push.aligned_resolution[1] = layout.aligned_height >> output_level;
			}
			else
			{
				input = wavelet.component_ll_views[component][output_level - 1];
			}

			push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
			push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

			[enc setComputePipelineState:encoder->device->dwt_pipeline[reads_input_plane ? 1 : 0]];
			[enc setBytes:&push length:sizeof(push) atIndex:0];
			[enc setTexture:input atIndex:0];
			[enc setTexture:wavelet.component_layer_views[component][output_level] atIndex:1];

			// Each threadgroup consumes a 32x32 source tile.
			[enc dispatchThreadgroups:MTLSizeMake((push.aligned_resolution[0] + 31) / 32,
					          (push.aligned_resolution[1] + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(DwtThreadgroupSize, 1, 1)];
		}
	}
}

// The remaining stages all walk the same (level, component, band) space as the
// decoder's dequant pass.
template <typename Op>
void for_each_band(const BlockLayout &layout, Op op)
{
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
				op(level, component, band);
		}
	}
}

void dispatch_quant(pyrowave_encoder encoder, id<MTLComputeCommandEncoder> enc)
{
	const auto &layout = encoder->layout;
	const int precision = encoder->device->precision;

	[enc setComputePipelineState:encoder->device->quant_pipeline];
	[enc setBuffer:encoder->meta_buffer offset:0 atIndex:1];
	[enc setBuffer:encoder->block_stat_buffer offset:0 atIndex:2];
	[enc setBuffer:encoder->payload_data offset:0 atIndex:3];
	[enc setSamplerState:encoder->device->border_sampler atIndex:0];

	for_each_band(layout, [&](int level, int component, int band) {
		QuantPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);
		push.input_layer = float(band);

		// The round trip through the 8 bit quant code is deliberate: the decoder
		// only ever sees the code, so quantize against the value it will dequantize
		// with rather than the ideal one.
		const float quant_res = get_quant_resolution(level, component, band, precision);
		push.quant_resolution = 1.0f / decode_quant(encode_quant(1.0f / quant_res));
		push.rdo_distortion_scale =
				get_quant_rdo_distortion_scale(level, component, band, precision, layout.chroma) *
				(1.0f / 256.0f);

		push.block_offset = layout.block_meta[component][level][band].block_offset_8x8;
		push.block_stride = layout.block_meta[component][level][band].block_stride_8x8;

		[enc setTexture:encoder->wavelet.component_layer_views[component][level] atIndex:0];
		[enc setBytes:&push length:sizeof(push) atIndex:0];
		[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 31) / 32,
		                                     (push.resolution[1] + 31) / 32, 1)
		      threadsPerThreadgroup:MTLSizeMake(QuantThreadgroupSize, 1, 1)];
	});
}

void dispatch_analyze_rdo(pyrowave_encoder encoder, id<MTLComputeCommandEncoder> enc)
{
	const auto &layout = encoder->layout;
	const int per_subdivision = compute_block_count_per_subdivision(layout.block_count_32x32);

	[enc setComputePipelineState:encoder->device->analyze_pipeline];
	[enc setBuffer:encoder->bucket_buffer offset:0 atIndex:0];
	[enc setBuffer:encoder->block_stat_buffer offset:0 atIndex:2];

	for_each_band(layout, [&](int level, int component, int band) {
		AnalyzePush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;

		const auto &meta = layout.block_meta[component][level][band];
		push.block_offset_8x8 = meta.block_offset_8x8;
		push.block_stride_8x8 = meta.block_stride_8x8;
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.total_wg_count = uint32_t(layout.block_count_32x32);
		push.num_blocks_aligned = uint32_t(per_subdivision * BlockSpaceSubdivision);
		push.block_index_shamt = floor_log2(uint32_t(per_subdivision));

		[enc setBytes:&push length:sizeof(push) atIndex:1];
		[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 31) / 32,
		                                     (push.resolution[1] + 31) / 32, 1)
		      threadsPerThreadgroup:MTLSizeMake(AnalyzeThreadgroupSize, 1, 1)];
	});

	// The finalize pass prefix sums the buckets the dispatches above filled.
	stage_barrier(enc, MTLBarrierScopeBuffers);
	[enc setComputePipelineState:encoder->device->analyze_finalize_pipeline];
	[enc setBuffer:encoder->bucket_buffer offset:0 atIndex:0];
	[enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(AnalyzeFinalizeThreadgroupSize, 1, 1)];
}

void dispatch_resolve_rdo(pyrowave_encoder encoder, id<MTLComputeCommandEncoder> enc, size_t target_size)
{
	const auto &layout = encoder->layout;

	// The sequence header is part of the frame's budget, so the payload target is
	// what is left after it. Guard the subtraction for absurdly small targets.
	if (target_size >= sizeof(BitstreamSequenceHeader))
		target_size -= sizeof(BitstreamSequenceHeader);

	ResolvePush push = {};
	push.target_payload_size = uint32_t(target_size / sizeof(uint32_t));
	push.num_blocks_per_subdivision = uint32_t(compute_block_count_per_subdivision(layout.block_count_32x32));

	[enc setComputePipelineState:encoder->device->resolve_pipeline];
	[enc setBuffer:encoder->bucket_buffer offset:0 atIndex:0];
	[enc setBytes:&push length:sizeof(push) atIndex:1];
	[enc setBuffer:encoder->quant_buffer offset:0 atIndex:2];
	[enc dispatchThreadgroups:MTLSizeMake(NumRDOBuckets * BlockSpaceSubdivision, 1, 1)
	      threadsPerThreadgroup:MTLSizeMake(ResolveThreadgroupSize, 1, 1)];
}

void dispatch_block_packing(pyrowave_encoder encoder, id<MTLComputeCommandEncoder> enc)
{
	const auto &layout = encoder->layout;
	const int precision = encoder->device->precision;

	// SPIRV-Cross renumbered these; the GLSL binding order is 1, 6, 4, 0, 5, 3.
	[enc setComputePipelineState:encoder->device->block_packing_pipeline];
	[enc setBuffer:encoder->payload_data offset:0 atIndex:0];
	[enc setBuffer:encoder->bitstream offset:0 atIndex:1];
	[enc setBuffer:encoder->quant_buffer offset:0 atIndex:3];
	[enc setBuffer:encoder->meta_buffer offset:0 atIndex:4];
	[enc setBuffer:encoder->block_stat_buffer offset:0 atIndex:5];
	[enc setBuffer:encoder->bitstream_meta offset:0 atIndex:6];

	for_each_band(layout, [&](int level, int component, int band) {
		BlockPackingPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_32x32_blocks[0] = (push.resolution[0] + 31) / 32;
		push.resolution_32x32_blocks[1] = (push.resolution[1] + 31) / 32;
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;

		// The code itself here, not the reciprocal the quantizer scales by.
		const float quant_res = get_quant_resolution(level, component, band, precision);
		push.quant_resolution_code = encode_quant(1.0f / quant_res);
		push.sequence_code = encoder->sequence_count;

		const auto &meta = layout.block_meta[component][level][band];
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.block_offset_8x8 = meta.block_offset_8x8;
		push.block_stride_8x8 = meta.block_stride_8x8;

		[enc setBytes:&push length:sizeof(push) atIndex:2];
		// Note the /2: unlike the other stages a threadgroup covers 2x2 of the
		// 32x32 blocks, one per 16 lanes.
		[enc dispatchThreadgroups:MTLSizeMake((push.resolution_32x32_blocks[0] + 1) / 2,
		                                     (push.resolution_32x32_blocks[1] + 1) / 2, 1)
		      threadsPerThreadgroup:MTLSizeMake(BlockPackingThreadgroupSize, 1, 1)];
	});
}

pyrowave_result encode_frame(pyrowave_encoder encoder, id<MTLTexture> const planes[NumComponents],
                             const pyrowave_rate_control *rate_control)
{
	// Block packing writes u32 words, so round the budget down like the Vulkan
	// encoder does.
	const size_t target_size = rate_control->maximum_bitstream_size & ~size_t(3);
	if (target_size == 0 || target_size > UINT32_MAX)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	// Same slack the Vulkan encoder leaves for the packer to overshoot into.
	const size_t meta_size = size_t(encoder->layout.block_count_32x32) * sizeof(BitstreamPacket);
	if (!ensure_bitstream_buffer(encoder, target_size + meta_size))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	encoder->sequence_count = (encoder->sequence_count + 1) & SequenceCountMask;

	id<MTLCommandBuffer> cmd = [encoder->queue commandBuffer];
	if (!cmd)
		return PYROWAVE_ERROR_GENERIC;
	cmd.label = @("pyrowave encode");

	auto blit = [cmd blitCommandEncoder];
	if (!blit)
		return PYROWAVE_ERROR_GENERIC;
	// Accumulated into, so they have to start from zero. The rest of the payload buffer
	// is fully overwritten by whatever is coded.
	[blit fillBuffer:encoder->payload_data range:NSMakeRange(0, 2 * sizeof(uint32_t)) value:0];
	[blit fillBuffer:encoder->bucket_buffer range:NSMakeRange(0, encoder->bucket_buffer.length) value:0];
	[blit fillBuffer:encoder->quant_buffer range:NSMakeRange(0, encoder->quant_buffer.length) value:0];
	// A block's payload rarely ends on a word boundary and the packer never writes the
	// leftover bytes of its final word, which still go out on the wire.
	//
	// This does NOT make the bitstream byte for byte reproducible, so never diff two
	// bitstreams to check a change -- diff the decoded output, which is deterministic.
	// The unused bits of a block's last sign byte come from shared_sign_bank in
	// block_packing.comp, which is never initialized and is updated with atomicAnd/Or
	// that preserve bits outside the write mask, so they are leftover threadgroup
	// memory. Decoders read only as many sign bits as there are significant
	// coefficients; the Vulkan encoder behaves the same way.
	[blit fillBuffer:encoder->bitstream range:NSMakeRange(0, target_size + meta_size) value:0];
	[blit endEncoding];

	// One concurrent encoder for the whole pipeline, with explicit barriers exactly where
	// the Vulkan encoder has them. Everything between two barriers is independent -- each
	// dispatch owns a distinct (component, level, band) region -- and a serial encoder
	// would barrier between all ~170.
	id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:
			bench_serial_dispatch() ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
	if (!enc)
		return PYROWAVE_ERROR_GENERIC;
	enc.label = @("pyrowave encode");

	dispatch_dwt(encoder, enc, planes);
	// The quantizer samples the coefficients the DWT just wrote.
	stage_barrier(enc, MTLBarrierScopeTextures);
	dispatch_quant(encoder, enc);
	// Rate control analysis reads the per block statistics and payload sizes.
	stage_barrier(enc, MTLBarrierScopeBuffers);
	dispatch_analyze_rdo(encoder, enc);
	stage_barrier(enc, MTLBarrierScopeBuffers);
	dispatch_resolve_rdo(encoder, enc, target_size);
	// Packing needs the quant decisions resolve just made.
	stage_barrier(enc, MTLBarrierScopeBuffers);
	dispatch_block_packing(encoder, enc);

	[enc endEncoding];
	[cmd commit];

	encoder->pending = cmd;
	encoder->have_result = true;

	return PYROWAVE_SUCCESS;
}

// Blocks until the last encode has landed, so the result buffers can be read.
pyrowave_result wait_for_result(pyrowave_encoder encoder)
{
	if (!encoder->have_result)
	{
		encoder->device->log("No frame has been encoded yet.");
		return PYROWAVE_ERROR_GENERIC;
	}

	if (encoder->pending)
	{
		[encoder->pending waitUntilCompleted];

#ifdef PYROWAVE_METAL_BENCH_HOOKS
		encoder->bench_last_gpu_ms =
				(encoder->pending.GPUEndTime - encoder->pending.GPUStartTime) * 1000.0;
#endif

		if (encoder->pending.status == MTLCommandBufferStatusError)
		{
			auto *error = encoder->pending.error;
			encoder->device->log("Encode command buffer failed: %s",
			                     error && error.localizedDescription ?
			                     error.localizedDescription.UTF8String : "unknown error");
			encoder->have_result = false;
			return PYROWAVE_ERROR_GENERIC;
		}

		encoder->pending = nullptr;
	}

	return PYROWAVE_SUCCESS;
}

// Validates the caller's CPU buffer and copies it into the input textures.
pyrowave_result upload_cpu_input(pyrowave_encoder encoder, const pyrowave_cpu_buffer *input)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int num_planes = input->format == PYROWAVE_CPU_BUFFER_FORMAT_NV12 ? 2 : 3;

	if (input->width != layout.width || input->height != layout.height)
	{
		device->log("Input is %dx%d, but the encoder was created for %dx%d.",
		            input->width, input->height, layout.width, layout.height);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	if (chroma_420 == (input->format == PYROWAVE_CPU_BUFFER_FORMAT_YUV444P))
	{
		device->log("Input format %d does not match the encoder's chroma subsampling.",
		            int(input->format));
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	for (int plane = 0; plane < num_planes; plane++)
	{
		const int plane_width = plane != 0 && chroma_420 ? layout.width / 2 : layout.width;
		const int plane_height = plane != 0 && chroma_420 ? layout.height / 2 : layout.height;
		// NV12 packs Cb and Cr into one 16bpp plane.
		const size_t bytes_per_pixel = num_planes == 2 && plane == 1 ? 2 : 1;

		if (!input->data[plane] ||
		    input->row_stride_in_bytes[plane] < size_t(plane_width) * bytes_per_pixel ||
		    input->row_stride_in_bytes[plane] * size_t(plane_height) > input->plane_size_in_bytes[plane])
		{
			device->log("Input plane %d is NULL or has an inconsistent stride and size.", plane);
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}
	}

	if (!ensure_cpu_input(encoder, input->format))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	// The textures are reused across frames, so the previous encode has to be done
	// reading them. In practice this is free: the packet queries have already
	// waited by the time a caller asks for another frame.
	if (encoder->pending)
		[encoder->pending waitUntilCompleted];

	// Both NV12 chroma views alias owned[1], so upload through the backing objects
	// rather than the sampled views.
	for (int plane = 0; plane < num_planes; plane++)
	{
		auto *texture = encoder->cpu_input.owned[plane];
		[texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
		           mipmapLevel:0
		             withBytes:input->data[plane]
		           bytesPerRow:input->row_stride_in_bytes[plane]];
	}

	return PYROWAVE_SUCCESS;
}

// Wraps the caller's IOSurfaces: either one biplanar NV12 surface, or three
// single plane R8 surfaces.
pyrowave_result wrap_gpu_input(pyrowave_encoder encoder, const pyrowave_gpu_input *input,
                               InputTextures &wrapped)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int chroma_width = chroma_420 ? layout.width / 2 : layout.width;
	const int chroma_height = chroma_420 ? layout.height / 2 : layout.height;

	auto surface = [&](int i) { return static_cast<IOSurfaceRef>(input->planes[i]); };

	if (!surface(0))
	{
		device->log("Input planes[0] is NULL.");
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	const bool biplanar = !surface(1) && !surface(2);
	if (!biplanar && (!surface(1) || !surface(2)))
	{
		device->log("Expected either one biplanar surface or three single plane surfaces.");
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	// Checks a surface plane against the geometry the shaders will read.
	auto plane_matches = [&](IOSurfaceRef surf, int plane, int width, int height) {
		if (int(IOSurfaceGetPlaneCount(surf)) <= plane && !(plane == 0 && IOSurfaceGetPlaneCount(surf) == 0))
		{
			device->log("Input surface has %zu planes, needs at least %d.",
			            size_t(IOSurfaceGetPlaneCount(surf)), plane + 1);
			return false;
		}

		// A non-planar surface reports 0 planes but still answers plane 0 queries.
		const int surface_width = int(IOSurfaceGetWidthOfPlane(surf, plane));
		const int surface_height = int(IOSurfaceGetHeightOfPlane(surf, plane));

		if (surface_width != width || surface_height != height)
		{
			device->log("Input surface plane %d is %dx%d, expected %dx%d.",
			            plane, surface_width, surface_height, width, height);
			return false;
		}

		return true;
	};

	if (biplanar)
	{
		if (!chroma_420)
		{
			device->log("Biplanar NV12 input has half resolution chroma, "
			            "so it cannot feed a 4:4:4 encoder.");
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}

		if (IOSurfaceGetPlaneCount(surface(0)) != 2)
		{
			device->log("Expected a biplanar surface in planes[0], but it has %zu planes.",
			            size_t(IOSurfaceGetPlaneCount(surface(0))));
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}

		if (!plane_matches(surface(0), 0, layout.width, layout.height) ||
		    !plane_matches(surface(0), 1, chroma_width, chroma_height))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;

		wrapped.sampled[0] = wrapped.adopt(wrap_surface_plane(
				device, surface(0), 0, MTLPixelFormatR8Unorm, layout.width, layout.height, false));

		auto *chroma = wrapped.adopt(wrap_surface_plane(
				device, surface(0), 1, MTLPixelFormatRG8Unorm, chroma_width, chroma_height, true));

		if (!wrapped.sampled[0] || !chroma)
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

		if (!make_interleaved_chroma_views(device, wrapped, chroma))
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
	}
	else
	{
		for (int i = 0; i < NumComponents; i++)
		{
			const int width = i == 0 ? layout.width : chroma_width;
			const int height = i == 0 ? layout.height : chroma_height;

			if (!plane_matches(surface(i), 0, width, height))
				return PYROWAVE_ERROR_INVALID_ARGUMENT;

			wrapped.sampled[i] = wrapped.adopt(wrap_surface_plane(
					device, surface(i), 0, MTLPixelFormatR8Unorm, width, height, false));

			if (!wrapped.sampled[i])
				return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
		}
	}

	return PYROWAVE_SUCCESS;
}

}

//////
// Public API

pyrowave_result pyrowave_encoder_create(const pyrowave_encoder_create_info *info,
                                        pyrowave_encoder *encoder)
{
	if (!info || !encoder || !info->device)
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


	// Deferred rather than done in pyrowave_device_create(), so that decode-only
	// users do not pay for six extra shader compiles.
	if (!ensure_encode_pipelines(info->device))
		return PYROWAVE_ERROR_SHADER_COMPILATION;

	auto created = std::unique_ptr<pyrowave_encoder_opaque>(new (std::nothrow) pyrowave_encoder_opaque);
	if (!created)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;

	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	created->queue = [created->device->mtl newCommandQueue];
	if (!created->queue)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	if (!created->wavelet.init(created->device, created->layout) ||
	    !create_encode_buffers(created.get()))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	*encoder = created.release();
	return PYROWAVE_SUCCESS;
}

void pyrowave_encoder_destroy(pyrowave_encoder encoder)
{
	delete encoder;
}

pyrowave_result pyrowave_encoder_encode_gpu_synchronous(pyrowave_encoder encoder,
                                                        const pyrowave_gpu_input *input,
                                                        const pyrowave_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;


	InputTextures wrapped;
	auto result = wrap_gpu_input(encoder, input, wrapped);
	if (result != PYROWAVE_SUCCESS)
		return result;

	// The command buffer retains the wrapping textures, so letting them go here is
	// safe even though the GPU has not run yet.
	return encode_frame(encoder, wrapped.sampled, rate_control);
}

pyrowave_result pyrowave_encoder_encode_cpu_synchronous(pyrowave_encoder encoder,
                                                        const pyrowave_cpu_buffer *input,
                                                        const pyrowave_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (input->format != PYROWAVE_CPU_BUFFER_FORMAT_NV12 &&
	    input->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV420P &&
	    input->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;


	auto result = upload_cpu_input(encoder, input);
	if (result != PYROWAVE_SUCCESS)
		return result;

	return encode_frame(encoder, encoder->cpu_input.sampled, rate_control);
}

pyrowave_result pyrowave_encoder_compute_num_packets(pyrowave_encoder encoder, size_t packet_boundary,
                                                     size_t *num_packets)
{
	if (!encoder || !num_packets || packet_boundary < sizeof(BitstreamSequenceHeader))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_SUCCESS)
		return result;

	*num_packets = compute_num_packets(encoder->layout, encoder->bitstream_meta.contents,
	                                   packet_boundary);
	return PYROWAVE_SUCCESS;
}

pyrowave_result pyrowave_encoder_packetize(pyrowave_encoder encoder, pyrowave_packet *packets,
                                           size_t packet_boundary, size_t *out_packets,
                                           void *bitstream, size_t size)
{
	if (!encoder || !packets || !out_packets || !bitstream ||
	    packet_boundary < sizeof(BitstreamSequenceHeader))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_SUCCESS)
		return result;

	static_assert(sizeof(pyrowave_packet) == sizeof(Packet), "pyrowave_packet layout mismatch.");
	*out_packets = packetize(encoder->layout, reinterpret_cast<Packet *>(packets), packet_boundary,
	                         bitstream, size,
	                         encoder->bitstream_meta.contents, encoder->bitstream.contents);
	return PYROWAVE_SUCCESS;
}

#ifdef PYROWAVE_METAL_BENCH_HOOKS
// Not in pyrowave_metal.h: the encoder owns its command buffer and deliberately does not
// hand it out, so a caller cannot time the GPU work itself. tools/metal/bench_encode.mm
// declares this extern directly.
extern "C" double pyrowave_bench_last_gpu_ms(pyrowave_encoder encoder)
{
	return encoder ? encoder->bench_last_gpu_ms : -1.0;
}
#endif
