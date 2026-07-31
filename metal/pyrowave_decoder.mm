// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Metal backend for the PyroWave decoder. Mirrors the compute path of
// pyrowave_decoder.cpp; the fragment iDWT path is not ported. The device object
// and the wavelet pyramid it shares with the encoder live in
// pyrowave_common.mm.

#include "pyrowave_common.hpp"

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

	[enc setComputePipelineState:decoder->device->dequant_pipeline];
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

void encode_idwt_dispatch(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc,
                          const IdwtPush &push, id<MTLTexture> input, id<MTLTexture> output,
                          bool dc_shift)
{
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
			// Final level writes the output planes directly. Under 420 the chroma
			// planes were already finished one level earlier.
			const int components = chroma_420 ? 1 : NumComponents;
			for (int c = 0; c < components; c++)
			{
				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->wavelet.component_layer_views[c][input_level],
				                     planes[c], true);
			}
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
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

	auto *cmd = (__bridge id<MTLCommandBuffer>)(command_buffer);

	// Every dequant dispatch writes a distinct (component, level, band) region of
	// the pyramid and none reads another's output, so they can all run at once. A
	// serial encoder would barrier between all ~42, but the cost is underutilization
	// rather than barrier latency: each dispatch is far too small to fill the GPU on
	// its own, which is why this pays at low resolution and not at 1080p 4:4:4.
	auto dequant_enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	if (!dequant_enc)
		return PYROWAVE_ERROR_GENERIC;

	dequant_enc.label = @("pyrowave dequant");
	encode_dequant(decoder, dequant_enc, slot);
	[dequant_enc endEncoding];

	// The iDWT is a dependent chain across levels, but the three components within
	// a level are independent, so this is also concurrent with explicit barriers
	// at the level boundaries only. Ordering against the dequant work above comes
	// from the encoder boundary, which Metal tracks automatically.
	auto idwt_enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	if (!idwt_enc)
		return PYROWAVE_ERROR_GENERIC;

	idwt_enc.label = @("pyrowave idwt");
	encode_idwt(decoder, idwt_enc, planes);
	[idwt_enc endEncoding];

	// Retained until this slot comes round again, so its buffers cannot be rewritten
	// while the GPU is still reading them.
	slot->consumer = cmd;

	decoder->parser.mark_frame_decoded();
	return PYROWAVE_SUCCESS;
}
