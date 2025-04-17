// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#ifndef DWT_COMMON_H_
#define DWT_COMMON_H_

const int APRON = 4;
const int APRON_HALF = APRON / 2;
const int BLOCK_SIZE = 32;
const int BLOCK_SIZE_HALF = BLOCK_SIZE >> 1;

const float ALPHA = -1.586134342059924;
const float BETA = -0.052980118572961;
const float GAMMA = 0.882911075530934;
const float DELTA = 0.443506852043971;
const float K = 1.230174104914001;
const float inv_K = 1.0 / K;

uint component;

#define FP32_SHARED 0

#if FP32_SHARED
shared float shared_block_x[gl_WorkGroupSize.y][BLOCK_SIZE + 2 * APRON][(BLOCK_SIZE + 2 * APRON) / 2 + 1];
shared float shared_block_y[gl_WorkGroupSize.y][BLOCK_SIZE + 2 * APRON][(BLOCK_SIZE + 2 * APRON) / 2 + 1];
vec2 load_shared_component(uint y, uint x, uint c) { return vec2(shared_block_x[c][y][x], shared_block_y[c][y][x]); }
void store_shared_component(uint y, uint x, uint c, vec2 v) { shared_block_x[c][y][x] = v.x; shared_block_y[c][y][x] = v.y; }
#else
shared uint shared_block[gl_WorkGroupSize.y][BLOCK_SIZE + 2 * APRON][(BLOCK_SIZE + 2 * APRON) / 2 + 1];
vec2 load_shared_component(uint y, uint x, uint c) { return unpackSnorm2x16(shared_block[c][y][x]); }
void store_shared_component(uint y, uint x, uint c, vec2 v) { shared_block[c][y][x] = packSnorm2x16(v); }
#endif

vec2 load_shared(uint y, uint x) { return load_shared_component(y, x, component); }
void store_shared(uint y, uint x, vec2 v) { store_shared_component(y, x, component, v); }

bvec2 band(bvec2 a, bvec2 b)
{
	return bvec2(a.x && b.x, a.y && b.y);
}

#include "dwt_swizzle.h"

#endif