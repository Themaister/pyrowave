// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#ifndef DWT_COMMON_H_
#define DWT_COMMON_H_

const int APRON = 4;
const int APRON_HALF = APRON / 2;
const int BLOCK_SIZE = 32;
const int BLOCK_SIZE_HALF = BLOCK_SIZE >> 1;

const float16_t ALPHA = -1.586134342059924hf;
const float16_t BETA = -0.052980118572961hf;
const float16_t GAMMA = 0.882911075530934hf;
const float16_t DELTA = 0.443506852043971hf;
const float16_t K = 1.230174104914001hf;
const float16_t inv_K = float16_t(1.0 / 1.230174104914001);

shared f16vec2 shared_block[(BLOCK_SIZE + 2 * APRON) / 2][(BLOCK_SIZE + 2 * APRON) + 1];
f16vec2 load_shared(uint y, uint x) { return shared_block[y][x]; }
void store_shared(uint y, uint x, f16vec2 v) { shared_block[y][x] = v; }

bvec2 band(bvec2 a, bvec2 b)
{
	return bvec2(a.x && b.x, a.y && b.y);
}

#include "dwt_swizzle.h"

#endif
