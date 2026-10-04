// SPDX-License-Identifier: MIT
#pragma once

#include "experimental_dequant.msl.h" // Shared unique-source replacement helper.
#include <stdint.h>

namespace PyroWave
{
inline bool build_reduced_barrier_idwt_msl(const char *canonical_source, int precision, std::string &source)
{
	// These fingerprints deliberately require the reviewed canonical shader
	// version. A transpiler or shader update needs a fresh synchronization audit;
	// matching a few surviving anchors alone would not prove the access regions.
	static const uint64_t reviewed_sources[] = {
		0xc81ec6deb0807faeull, // FP16 math/storage.
		0x06519ddf727250c7ull, // FP32 math, FP16 storage.
		0x948127a862a5467bull  // FP32 math/storage.
	};
	if (precision < 0 || precision > 2)
		return false;
	uint64_t fingerprint = 14695981039346656037ull;
	for (auto *p = reinterpret_cast<const unsigned char *>(canonical_source); *p; p++)
		fingerprint = (fingerprint ^ *p) * 1099511628211ull;
	if (fingerprint != reviewed_sources[precision])
		return false;

	// Coordinate proof for the fixed 64-thread caller:
	// - inverse_transform8x2 reads rows 0..15 / columns 0..39 and crosses its
	//   WAR barrier before writing rows 0..15 / columns 0..31.
	// - inverse_transform4x2(index < 32, y_offset = 16) reads rows 16..19 /
	//   columns 0..39 and writes rows 0..15 / columns 32..39.
	// The apron read/write regions are disjoint, and its writes cannot race with
	// the first helper's reads (already fenced) or writes (different columns).
	// The kernel's next barrier still joins both before the vertical transform.
	// No arithmetic, scratch representation, or other barrier changes.
	source = canonical_source;
	return replace_unique_msl(source, "kernel void pyrowave_idwt(", "kernel void pyrowave_idwt_reduced_barriers(") &&
	       replace_unique_msl(source,
		"    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (active_lane)\n    {\n        for (int i_5 = 2; i_5 < 4; i_5++)",
		"    }\n    if (active_lane)\n    {\n        for (int i_5 = 2; i_5 < 4; i_5++)");
}
}
