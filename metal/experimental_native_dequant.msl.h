// SPDX-License-Identifier: MIT
#pragma once

// Private native unpacking experiment. Only the integer magnitude decoder is
// replaced; the reviewed canonical shader retains all dispatch, sparse-block,
// sign scan, scaling and texture-write behavior.
#include "shaders/pyrowave_msl.h"
#include <stdint.h>
#include <string>

namespace PyroWave
{
inline const std::string &native_dequant_source()
{
	static const std::string source = [] {
		uint64_t fingerprint = 14695981039346656037ull;
		for (auto *p = reinterpret_cast<const unsigned char *>(wavelet_dequant_msl_source); *p; p++)
			fingerprint = (fingerprint ^ *p) * 1099511628211ull;
		// Fail closed after any canonical update until its bitstream contract has
		// been reviewed again. An empty source means this experiment is unavailable.
		if (fingerprint != 0xd6a10cc9a170610eull)
			return std::string{};

		std::string result = wavelet_dequant_msl_source;
		const char *begin_anchor = "static inline __attribute__((always_inline))\nfloat2x4 decode_payload(";
		const char *end_anchor = "static inline __attribute__((always_inline))\nfloat decode_quant(";
		const size_t begin = result.find(begin_anchor);
		const size_t end = result.find(end_anchor, begin);
		if (begin == std::string::npos || end == std::string::npos ||
		    result.find(begin_anchor, begin + 1) != std::string::npos ||
		    result.find(end_anchor, end + 1) != std::string::npos)
			return std::string{};

		result.replace(begin, end - begin, R"PYROWAVE_NATIVE(
// Transpose eight packed bit-plane bytes into eight coefficient bytes. Use two
// 32-bit words rather than requiring 64-bit integer ALU operations. After these
// three butterfly stages, output byte b contains input bit b from each plane.
static inline __attribute__((always_inline))
uint2 native_transpose_planes(uint2 planes)
{
    uint t0 = (planes.x ^ ((planes.x >> 7u) | (planes.y << 25u))) & 0x00aa00aau;
    uint t1 = (planes.y ^ (planes.y >> 7u)) & 0x00aa00aau;
    planes.x ^= t0 ^ (t0 << 7u);
    planes.y ^= t1 ^ (t1 << 7u);

    t0 = (planes.x ^ ((planes.x >> 14u) | (planes.y << 18u))) & 0x0000ccccu;
    t1 = (planes.y ^ (planes.y >> 14u)) & 0x0000ccccu;
    planes.x ^= t0 ^ (t0 << 14u);
    planes.y ^= t1 ^ (t1 << 14u);

    t0 = (planes.x ^ ((planes.x >> 28u) | (planes.y << 4u))) & 0xf0f0f0f0u;
    planes.x ^= t0;
    planes.y ^= t0 >> 4u;
    return planes;
}

// The buffer base is naturally uint-aligned. Plane byte offsets are not: read
// aligned words and assemble the two words rather than casting an unaligned
// plane pointer. The decoder already provides sixteen trailing padding bytes;
// an eight-byte chunk can read at most ten bytes past its final useful plane.
static inline __attribute__((always_inline))
uint2 native_read_plane_chunk(const device Payloads8& payload, uint byte_offset)
{
    const device uint* words = (const device uint*)payload.data;
    uint word_offset = byte_offset >> 2u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = words[word_offset];
    uint hi = words[word_offset + 1u];
    if (shift != 0u)
    {
        uint next = words[word_offset + 2u];
        lo = (lo >> shift) | (hi << (32u - shift));
        hi = (hi >> shift) | (next << (32u - shift));
    }
    return uint2(lo, hi);
}

static inline __attribute__((always_inline))
float2x4 decode_payload(thread const uint& code_word, thread const uint& q_bits,
                       thread const uint& offset, thread const uint& block_index,
                       const device Payloads8& payload_data_u8)
{
    if (code_word == 0u)
        return float2x4(float4(0.0), float4(0.0));

    uint bit_offset = 2u * block_index;
    uint prefix_mask = (1u << bit_offset) - 1u;
    uint lsbs = code_word & 0x5555u;
    uint msbs = code_word & 0xaaaau;
    msbs |= msbs >> 1u;
    uint byte_offset = popcount(lsbs & prefix_mask) + popcount(msbs & prefix_mask) +
                       q_bits * block_index + offset;
    uint remaining = q_bits + ((code_word >> bit_offset) & 3u);
    uint4 decoded_even = uint4(0u);
    uint4 decoded_odd = uint4(0u);

    while (remaining != 0u)
    {
        uint count = min(remaining, 8u);
        uint2 chunk = native_transpose_planes(native_read_plane_chunk(payload_data_u8, byte_offset));
        // Reversing each word reverses plane order inside each byte and byte
        // order inside the word. Select bytes in reverse order to undo the
        // latter, preserving coefficient order 0,2,4,6 / 1,3,5,7.
        chunk = uint2(reverse_bits(chunk.x), reverse_bits(chunk.y));
        uint4 even = uint4(chunk.x >> 24u, chunk.x >> 8u, chunk.y >> 24u, chunk.y >> 8u) & 255u;
        uint4 odd = uint4(chunk.x >> 16u, chunk.x, chunk.y >> 16u, chunk.y) & 255u;
        // The final short chunk occupies its most significant byte bits.
        // Discard bytes belonging to the next 4x2 block or packed sign stream.
        even >>= 8u - count;
        odd >>= 8u - count;
        decoded_even = (decoded_even << count) | even;
        decoded_odd = (decoded_odd << count) | odd;
        byte_offset += 8u;
        remaining -= count;
    }

    float4 even = float4(decoded_even);
    float4 odd = float4(decoded_odd);
    even += select(float4(0.0), float4(0.5), decoded_even != uint4(0u));
    odd += select(float4(0.0), float4(0.5), decoded_odd != uint4(0u));
    return float2x4(even, odd);
}

)PYROWAVE_NATIVE");
		return result;
	}();
	return source;
}

// Separate hybrid experiment: keep native_dequant_source() byte-for-byte stable
// while avoiding its full 8x8 transpose for the common short-plane payloads.
inline const std::string &hybrid_dequant_source()
{
	static const std::string source = [] {
		std::string result = native_dequant_source();
		if (result.empty())
			return result;
		const char *anchor = "    while (remaining != 0u)\n";
		const size_t position = result.find(anchor);
		if (position == std::string::npos || result.find(anchor, position + 1) != std::string::npos)
			return std::string{};
		result.insert(position, R"PYROWAVE_HYBRID(
    if (remaining == 1u)
    {
        uint plane = uint(payload_data_u8.data[byte_offset]);
        decoded_even = (uint4(plane) >> uint4(0u, 2u, 4u, 6u)) & 1u;
        decoded_odd = (uint4(plane) >> uint4(1u, 3u, 5u, 7u)) & 1u;
        remaining = 0u;
    }
    else if (remaining >= 2u && remaining <= 4u)
    {
        const device uint* words = (const device uint*)payload_data_u8.data;
        uint word_offset = byte_offset >> 2u;
        uint shift = (byte_offset & 3u) * 8u;
        uint planes = words[word_offset];
        if (shift != 0u)
            planes = (planes >> shift) | (words[word_offset + 1u] << (32u - shift));

        // Each selected coefficient bit lies in the low bit of a byte.
        // Multiplication gathers four such bits into the high-byte nibble:
        // plane0*8 + plane1*4 + plane2*2 + plane3. No byte can carry.
        uint4 even = ((uint4(planes) >> uint4(0u, 2u, 4u, 6u)) & 0x01010101u) * 0x08040201u;
        uint4 odd = ((uint4(planes) >> uint4(1u, 3u, 5u, 7u)) & 0x01010101u) * 0x08040201u;
        decoded_even = even >> (28u - remaining);
        decoded_odd = odd >> (28u - remaining);
        remaining = 0u;
    }
)PYROWAVE_HYBRID");
		return result;
	}();
	return source;
}
}
