// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#include "pyrowave_bitstream.hpp"
#include <algorithm>
#include <stdio.h>

// Granite's LOGE is not available here; the CPU bitstream path is deliberately
// dependency free, so errors go straight to stderr.
#define PYROWAVE_LOGE(...) fprintf(stderr, "pyrowave: " __VA_ARGS__)

namespace PyroWave
{
int BlockLayout::level_width(int level) const
{
	return std::max<int>(1, (aligned_width / 2) >> level);
}

int BlockLayout::level_height(int level) const
{
	return std::max<int>(1, (aligned_height / 2) >> level);
}

void BlockLayout::accumulate_block_mapping(int blocks_x_8x8, int blocks_y_8x8)
{
	int blocks_x_32x32 = (blocks_x_8x8 + 3) / 4;
	int blocks_y_32x32 = (blocks_y_8x8 + 3) / 4;

	for (int y = 0; y < blocks_y_32x32; y++)
	{
		for (int x = 0; x < blocks_x_32x32; x++)
		{
			BlockMapping mapping = {};
			mapping.block_offset_8x8 = block_count_8x8 + 4 * y * blocks_x_8x8 + 4 * x;
			mapping.block_stride_8x8 = blocks_x_8x8;
			mapping.block_width_8x8 = std::min<int>(4, blocks_x_8x8 - 4 * x);
			mapping.block_height_8x8 = std::min<int>(4, blocks_y_8x8 - 4 * y);
			block_32x32_to_8x8_mapping.push_back(mapping);
			block_count_32x32++;
		}
	}

	block_count_8x8 += blocks_x_8x8 * blocks_y_8x8;
}

bool BlockLayout::init(int width_, int height_, ChromaSubsampling chroma_)
{
	// width_minus_1 / height_minus_1 are 14 bits in the sequence header.
	if (width_ <= 0 || height_ <= 0 || width_ > 16384 || height_ > 16384)
	{
		PYROWAVE_LOGE("Dimensions (%d, %d) are out of range.\n", width_, height_);
		return false;
	}

	width = width_;
	height = height_;
	chroma = chroma_;

	aligned_width = align(width, Alignment);
	aligned_height = align(height, Alignment);
	aligned_width = std::max<int>(aligned_width, MinimumImageSize);
	aligned_height = std::max<int>(aligned_height, MinimumImageSize);

	block_32x32_to_8x8_mapping.clear();
	block_count_8x8 = 0;
	block_count_32x32 = 0;

	for (int level = DecompositionLevels - 1; level >= 0; level--)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				int level_w = level_width(level);
				int level_h = level_height(level);

				int blocks_x_8x8 = (level_w + 7) / 8;
				int blocks_y_8x8 = (level_h + 7) / 8;
				int blocks_x_32x32 = (level_w + 31) / 32;
				int blocks_y_32x32 = (level_h + 31) / 32;

				block_meta[component][level][band] = {
					block_count_8x8, blocks_x_8x8,
					block_count_32x32, blocks_x_32x32,
					blocks_x_32x32 * blocks_y_32x32
				};

				accumulate_block_mapping(blocks_x_8x8, blocks_y_8x8);
			}
		}
	}

	return true;
}

void BitstreamParser::init(const BlockLayout *layout_)
{
	layout = layout_;
	dequant_offset_buffer_cpu.resize(layout->block_count_32x32);
	payload_data_cpu.reserve(1024 * 1024);
	clear();
}

void BitstreamParser::clear()
{
	std::fill(dequant_offset_buffer_cpu.begin(), dequant_offset_buffer_cpu.end(), UINT32_MAX);
	decoded_blocks = 0;
	last_seq = UINT32_MAX;
	decoded_frame_for_current_sequence = false;
	total_blocks_in_sequence = layout->block_count_32x32;
	payload_data_cpu.clear();
}

bool BitstreamParser::decode_packet(const BitstreamHeader *header)
{
	auto &offset = dequant_offset_buffer_cpu[header->block_index];
	if (offset == UINT32_MAX)
	{
		decoded_blocks++;
		offset = (unsigned int)payload_data_cpu.size();
	}
	else
	{
		return true;
	}

	auto *payload_words = reinterpret_cast<const uint32_t *>(header);

	if (sizeof(*header) / sizeof(uint32_t) > header->payload_words)
	{
		PYROWAVE_LOGE("payload_words is not large enough.\n");
		return false;
	}

	payload_data_cpu.insert(
			payload_data_cpu.end(),
			payload_words,
			payload_words + header->payload_words);
	return true;
}

bool BitstreamParser::push_packet(const void *data_, size_t size)
{
	auto *data = static_cast<const uint8_t *>(data_);
	while (size >= sizeof(BitstreamHeader))
	{
		auto *header = reinterpret_cast<const BitstreamHeader *>(data);

		if (header->extended != 0)
		{
			auto *seq = reinterpret_cast<const BitstreamSequenceHeader *>(header);

			if (sizeof(*header) > size)
			{
				PYROWAVE_LOGE("Parsing sequence header, but only %zu bytes left to parse.\n", size);
				return false;
			}

			if (seq->chroma_resolution != int(layout->chroma))
			{
				PYROWAVE_LOGE("Chroma resolution mismatch!\n");
				return false;
			}

			uint8_t diff = (header->sequence - last_seq) & SequenceCountMask;
			if (last_seq != UINT32_MAX && diff > (SequenceCountMask / 2))
			{
				return true;
			}

			if (last_seq == UINT32_MAX || diff != 0)
			{
				clear();
				last_seq = header->sequence;
			}

			if (seq->code == BITSTREAM_EXTENDED_CODE_START_OF_FRAME)
			{
				if (seq->width_minus_1 + 1 != uint32_t(layout->width) ||
				    seq->height_minus_1 + 1 != uint32_t(layout->height))
				{
					PYROWAVE_LOGE("Dimension mismatch in seq packet, (%u, %u) != (%d, %d)\n",
					              seq->width_minus_1 + 1, seq->height_minus_1 + 1, layout->width, layout->height);
					return false;
				}

				total_blocks_in_sequence = int(seq->total_blocks);
			}
			else
			{
				PYROWAVE_LOGE("Unrecognized sequence header mode %u.\n", seq->code);
				return false;
			}

			data += sizeof(*header);
			size -= sizeof(*header);

			continue;
		}

		size_t packet_size = header->payload_words * sizeof(uint32_t);

		if (packet_size > size)
		{
			PYROWAVE_LOGE("Packet header states %zu bytes, but only %zu bytes left to parse.\n", packet_size, size);
			return false;
		}

		bool restart;

		if (last_seq == UINT32_MAX)
		{
			restart = true;
		}
		else
		{
			uint8_t diff = (header->sequence - last_seq) & SequenceCountMask;
			if (diff > (SequenceCountMask / 2))
			{
				return true;
			}
			restart = diff != 0;
		}

		if (restart)
		{
			clear();
			last_seq = header->sequence;
		}

		if (header->block_index >= uint32_t(layout->block_count_32x32))
		{
			PYROWAVE_LOGE("block_index %u is out of bounds (>= %d).\n",
			              header->block_index, layout->block_count_32x32);
			return false;
		}

		if (!decode_packet(header))
			return false;

		data += packet_size;
		size -= packet_size;
	}

	if (size != 0)
	{
		PYROWAVE_LOGE("Did not consume packet completely.\n");
		return false;
	}

	return true;
}

bool BitstreamParser::has_pristine_bands(int bands, const uint32_t *active_block_mask, size_t word_count) const
{
	// Account for 4:2:0 where level0 will not have packets.
	// It's somewhat meaningless to ask for pristine bands all the way up to that point though.
	assert(bands < DecompositionLevels);

	// This analysis assumes that there are no "null" blocks present.
	// For the lowest frequency bands, that is vanishingly unlikely to happen,
	// and worst case we get a false positive rejection.
	// The encoder's compute_block_active_words() can supply the real mask as
	// sideband data when that matters.

	const auto block_is_missing = [&](uint32_t block_index)
	{
		if (dequant_offset_buffer_cpu[block_index] != UINT32_MAX)
			return false;

		uint32_t word_index = block_index / 32;

		if (!active_block_mask || word_index >= word_count)
			return true;

		// If the block wasn't expected to be active anyway, just pass it through.
		return ((active_block_mask[word_index] >> (block_index % 32)) & 1) != 0;
	};

	for (int band = 0; band < bands; band++)
	{
		for (auto &component : layout->block_meta)
		{
			if (band == 0)
			{
				auto &meta = component[DecompositionLevels - 1][0];
				for (int i = 0; i < meta.block_count_32x32; i++)
					if (block_is_missing(meta.block_offset_32x32 + i))
						return false;
			}
			else
			{
				// If we can reconstruct the LH, HL, HH bands, we can generate the higher-resolution LL band.
				for (int high_freq_bands = 1; high_freq_bands < 4; high_freq_bands++)
				{
					auto &meta = component[DecompositionLevels - band][high_freq_bands];
					for (int i = 0; i < meta.block_count_32x32; i++)
						if (block_is_missing(meta.block_offset_32x32 + i))
							return false;
				}
			}
		}
	}

	return true;
}

bool BitstreamParser::decode_is_ready(bool allow_partial_frame, int num_pristine_bands, float minimum_packet_ratio,
                                      const uint32_t *active_block_mask, size_t word_count) const
{
	if (decoded_frame_for_current_sequence)
		return false;

	if (last_seq == UINT32_MAX)
		return false;

	if (decoded_blocks < total_blocks_in_sequence)
	{
		if (!allow_partial_frame)
			return false;

		if (!has_pristine_bands(num_pristine_bands, active_block_mask, word_count))
			return false;
		if (float(decoded_blocks) <= float(total_blocks_in_sequence) * minimum_packet_ratio)
			return false;
	}

	return true;
}

bool BitstreamParser::decode_is_ready(bool allow_partial_frame) const
{
	// At the very least, we want some LL bands to be received properly,
	// otherwise we get extreme artifacts.
	return decode_is_ready(allow_partial_frame, 2, 0.9f, nullptr, 0);
}

void BitstreamParser::mark_frame_decoded()
{
	decoded_frame_for_current_sequence = true;
}

//////
// Encoder side.

int compute_block_count_per_subdivision(int num_blocks)
{
	int per_subdivision = align(num_blocks, BlockSpaceSubdivision) / BlockSpaceSubdivision;

	// Round up to a power of two.
	int pot = 1;
	while (pot < per_subdivision)
		pot *= 2;

	return pot;
}

size_t get_num_active_blocks(const BlockLayout &layout, int bands)
{
	assert(bands < DecompositionLevels);
	if (bands <= 0)
		return 0;

	int last_subband = bands == 1 ? 0 : 3;
	int last_band = bands - 1;

	auto &meta = layout.block_meta[NumComponents - 1][DecompositionLevels - std::max<int>(1, last_band)][last_subband];
	return meta.block_offset_32x32 + meta.block_count_32x32;
}

void compute_block_active_words(const BlockLayout &layout, int bands, uint32_t *words, size_t word_count,
                                const void *mapped_meta)
{
	auto *meta = static_cast<const BitstreamPacket *>(mapped_meta);
	memset(words, 0, sizeof(uint32_t) * word_count);

	size_t num_active_blocks = get_num_active_blocks(layout, bands);
	assert(word_count * 32 >= num_active_blocks);

	for (size_t i = 0; i < num_active_blocks; i++)
		if (meta[i].num_words)
			words[i / 32] |= 1u << (i % 32);
}

size_t compute_num_critical_packets(const BlockLayout &layout, int bands, const void *mapped_meta,
                                    size_t packet_boundary, size_t padding_size)
{
	auto *meta = static_cast<const BitstreamPacket *>(mapped_meta);
	size_t num_packets = 0;
	size_t size_in_packet = 0;

	size_in_packet += sizeof(BitstreamSequenceHeader);

	int block_count = bands >= 0 ? int(get_num_active_blocks(layout, bands)) : layout.block_count_32x32;

	for (int i = 0; i < block_count; i++)
	{
		size_t packet_size = meta[i].num_words * sizeof(uint32_t);
		if (!packet_size)
			continue;

		if (size_in_packet + packet_size + padding_size > packet_boundary)
		{
			size_in_packet = 0;
			padding_size = 0;
			num_packets++;
		}

		size_in_packet += packet_size;
	}

	if (size_in_packet)
		num_packets++;

	return num_packets;
}

size_t compute_num_packets(const BlockLayout &layout, const void *mapped_meta, size_t packet_boundary,
                           size_t padding_size)
{
	return compute_num_critical_packets(layout, -1, mapped_meta, packet_boundary, padding_size);
}

size_t packetize(const BlockLayout &layout, Packet *packets, size_t packet_boundary,
                 void *output_bitstream_, size_t size,
                 const void *mapped_meta, const void *mapped_bitstream,
                 size_t padding_size)
{
	size_t num_packets = 0;
	size_t size_in_packet = 0;
	size_t packet_offset = 0;
	size_t output_offset = 0;
	auto *meta = static_cast<const BitstreamPacket *>(mapped_meta);
	auto *input_bitstream = static_cast<const uint32_t *>(mapped_bitstream);
	auto *output_bitstream = static_cast<uint8_t *>(output_bitstream_);

	size_t num_non_zero_blocks = 0;
	for (int i = 0; i < layout.block_count_32x32; i++)
		if (meta[i].num_words != 0)
			num_non_zero_blocks++;

	BitstreamSequenceHeader header = {};
	header.width_minus_1 = layout.width - 1;
	header.height_minus_1 = layout.height - 1;
	header.sequence = reinterpret_cast<const BitstreamHeader *>(input_bitstream + meta[0].offset_u32)->sequence;
	header.extended = 1;
	header.code = BITSTREAM_EXTENDED_CODE_START_OF_FRAME;
	header.total_blocks = uint32_t(num_non_zero_blocks);
	header.chroma_resolution = layout.chroma == ChromaSubsampling::Chroma444 ?
	                           CHROMA_RESOLUTION_444 : CHROMA_RESOLUTION_420;

	if (sizeof(header) > size)
		return 0;

	memcpy(output_bitstream, &header, sizeof(header));
	output_offset += sizeof(header);
	size_in_packet += sizeof(header);

	for (int i = 0; i < layout.block_count_32x32; i++)
	{
		size_t packet_size = meta[i].num_words * sizeof(uint32_t);
		if (!packet_size)
			continue;

		if (size_in_packet + packet_size + padding_size > packet_boundary)
		{
			packets[num_packets++] = { packet_offset, size_in_packet };
			size_in_packet = 0;
			packet_offset = output_offset;
			padding_size = 0;
		}

		if (output_offset + packet_size > size)
			return num_packets;

		memcpy(output_bitstream + output_offset, input_bitstream + meta[i].offset_u32, packet_size);

		output_offset += packet_size;
		size_in_packet += packet_size;
	}

	if (size_in_packet)
		packets[num_packets++] = { packet_offset, size_in_packet };

	return num_packets;
}
}
