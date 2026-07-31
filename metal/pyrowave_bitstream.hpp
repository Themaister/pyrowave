// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

// Granite-free CPU side of the PyroWave decoder: bitstream layout, block metadata
// and packet parsing. Contains no graphics API dependency, so it can back either
// the Vulkan decoder or the Metal one.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <vector>

namespace PyroWave
{
enum class ChromaSubsampling
{
	Chroma420,
	Chroma444
};
struct BitstreamHeader
{
	uint16_t ballot;
	uint16_t payload_words : 12;
	uint16_t sequence : 3;
	uint16_t extended : 1;
	uint32_t quant_code : 8;
	uint32_t block_index : 24;
};

static_assert(sizeof(BitstreamHeader) == 8, "BitstreamHeader is not 8 bytes.");

struct BitstreamSequenceHeader
{
	uint32_t width_minus_1 : 14;
	uint32_t height_minus_1 : 14;
	uint32_t sequence : 3;
	uint32_t extended : 1;
	uint32_t total_blocks : 24;
	uint32_t code : 2;
	uint32_t chroma_resolution : 1;
	uint32_t color_primaries : 1;
	uint32_t transfer_function : 1;
	uint32_t ycbcr_transform : 1;
	uint32_t ycbcr_range : 1;
	uint32_t chroma_siting : 1;
};

static_assert(sizeof(BitstreamSequenceHeader) == 8, "BitstreamSequenceHeader is not 8 bytes.");

// Written by the encoder's block packing pass, one per 32x32 block. A zero
// num_words means the block was not coded.
struct BitstreamPacket
{
	uint32_t offset_u32;
	uint32_t num_words;
};

enum
{
	BITSTREAM_EXTENDED_CODE_START_OF_FRAME = 0,
};

enum
{
	CHROMA_RESOLUTION_420 = 0,
	CHROMA_RESOLUTION_444 = 1
};

static constexpr uint32_t SequenceCountMask = 0x7;

static constexpr int DecompositionLevels = 5;
static constexpr int Alignment = 1 << DecompositionLevels;
// If the final decomposition band is too small, the mirroring will break since it starts double mirroring.
static constexpr int MinimumImageSize = 4 << DecompositionLevels;
static constexpr int NumComponents = 3;
static constexpr int NumFrequencyBandsPerLevel = 4;

static inline int align(int value, int alignment)
{
	return (value + alignment - 1) & ~(alignment - 1);
}

// The 8 bit quantization code carried in BitstreamHeader::quant_code. A custom
// float formulation for numbers in the (0, 2) range; both directions are part of
// the bitstream definition, so they live here rather than in either backend.
static constexpr int MaxScaleExp = 4;

static inline float decode_quant(uint8_t quant_code)
{
	int e = MaxScaleExp - (quant_code >> 3);
	int m = quant_code & 0x7;
	return (1.0f / (8.0f * 1024.0f * 1024.0f)) * float((8 + m) * (1 << (20 + e)));
}

static inline uint8_t encode_quant(float decoder_q_scale)
{
	uint32_t v;
	memcpy(&v, &decoder_q_scale, sizeof(decoder_q_scale));

	int e = ((v >> 23) & 0xff) - 127 - MaxScaleExp;
	int m = (v >> 20) & 0x7;
	e = -e;
	assert(e >= 0 && e <= 20);
	return uint8_t((e << 3) | m);
}

// Pure integer derivation of where every 8x8 and 32x32 block lives in the wavelet
// image pyramid. Mirrors WaveletBuffers::init_block_meta(), but derives the mip
// dimensions arithmetically instead of querying a Vulkan image.
struct BlockLayout
{
	// Returns false if the requested dimensions cannot be represented in the bitstream.
	bool init(int width, int height, ChromaSubsampling chroma);

	struct BlockInfo
	{
		int block_offset_8x8;
		int block_stride_8x8;
		int block_offset_32x32;
		int block_stride_32x32;
	};
	BlockInfo block_meta[NumComponents][DecompositionLevels][NumFrequencyBandsPerLevel] = {};

	struct BlockMapping
	{
		int block_offset_8x8;
		int block_stride_8x8;
		int block_width_8x8;
		int block_height_8x8;
	};
	std::vector<BlockMapping> block_32x32_to_8x8_mapping;

	int block_count_8x8 = 0;
	int block_count_32x32 = 0;

	int width = 0;
	int height = 0;
	int aligned_width = 0;
	int aligned_height = 0;

	ChromaSubsampling chroma = ChromaSubsampling::Chroma420;

	// Dimensions of mip `level` of the wavelet image, which is allocated at half
	// the aligned frame size. Matches Vulkan::Image::get_width/get_height(lod).
	int level_width(int level) const;
	int level_height(int level) const;

private:
	void accumulate_block_mapping(int blocks_x_8x8, int blocks_y_8x8);
};

// Parses packets into the two flat arrays the dequant shader consumes:
// a per-32x32-block offset table and the concatenated payload words.
class BitstreamParser
{
public:
	// `layout` must outlive the parser.
	void init(const BlockLayout *layout);

	void clear();

	bool push_packet(const void *data, size_t size);

	bool decode_is_ready(bool allow_partial_frame) const;

	// Call once a frame has actually been submitted for decode, so the same
	// sequence is not decoded twice.
	void mark_frame_decoded();

	// UINT32_MAX marks a block that never received a packet. Indexed by 32x32 block
	// index, sized block_count_32x32. Upload verbatim to the dequant offset buffer.
	const std::vector<uint32_t> &dequant_offsets() const { return dequant_offset_buffer_cpu; }

	// Concatenated packet payloads, indexed by the offsets above. Note the dequant
	// shader can read slightly past the end, so the GPU buffer needs padding.
	const std::vector<uint32_t> &payload() const { return payload_data_cpu; }

	int get_decoded_blocks() const { return decoded_blocks; }
	int get_total_blocks_in_sequence() const { return total_blocks_in_sequence; }

private:
	const BlockLayout *layout = nullptr;

	std::vector<uint32_t> dequant_offset_buffer_cpu;
	std::vector<uint32_t> payload_data_cpu;
	int decoded_blocks = 0;
	int total_blocks_in_sequence = 0;
	uint32_t last_seq = UINT32_MAX;
	bool decoded_frame_for_current_sequence = false;

	bool decode_packet(const BitstreamHeader *header);
};

//////
// Encoder side. Turning the GPU's per block output into network packets is pure
// CPU work with no graphics dependency, so it lives here alongside the parser.

// Sizes of the encoder's GPU side scratch buffers. Only here because they are
// pure layout, derived from the block counts.
static constexpr int BlockSpaceSubdivision = 16;
static constexpr int NumRDOBuckets = 128;
static constexpr int RDOBucketOffset = 64;

struct BlockStats
{
	uint16_t square_error_fp16;
	uint16_t encode_cost_bits;
};

struct BlockStatsBlock
{
	uint32_t num_planes;
	BlockStats stats[15];
};
static_assert(sizeof(BlockStatsBlock) == 64, "BlockStatsBlock is not 64 bytes.");

struct BlockMeta
{
	uint32_t code_word;
	uint32_t offset;
};

struct RDOperation
{
	int32_t quant;
	uint16_t block_offset;
	uint16_t block_saving;
};

int compute_block_count_per_subdivision(int num_blocks);

struct Packet
{
	size_t offset;
	size_t size;
};

// How many packets the frame needs if each may carry at most packet_boundary
// bytes. `mapped_meta` is block_count_32x32 BitstreamPacket entries.
size_t compute_num_packets(const BlockLayout &layout, const void *mapped_meta, size_t packet_boundary);

// Copies the coded blocks into `output_bitstream`, prefixed by a sequence header,
// and fills in the packet boundaries. Returns the number of packets written,
// which is at most what compute_num_packets() reported.
size_t packetize(const BlockLayout &layout, Packet *packets, size_t packet_boundary,
                 void *output_bitstream, size_t size,
                 const void *mapped_meta, const void *mapped_bitstream);
}
