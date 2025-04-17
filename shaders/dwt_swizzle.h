// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#ifndef DWT_SWIZZLE_H_
#define DWT_SWIZZLE_H_

ivec2 unswizzle16x8(uint index)
{
	uint y = bitfieldExtract(index, 0, 1);
	uint x = bitfieldExtract(index, 1, 2);
	y |= bitfieldExtract(index, 3, 2) << 1;
	x |= bitfieldExtract(index, 5, 2) << 2;
	return ivec2(x, y);
}

ivec2 unswizzle16x16_dequant(uint index)
{
	uint x = bitfieldExtract(index, 0, 1);
	uint y = bitfieldExtract(index, 1, 2);
	x |= bitfieldExtract(index, 3, 2) << 1;
	y |= bitfieldExtract(index, 5, 2) << 2;
	x |= bitfieldExtract(index, 7, 1) << 3;
	return ivec2(x, y);
}

ivec2 unswizzle8x8_2x2_quant(uint index)
{
	uint y = bitfieldExtract(index, 0, 1);
	uint x = bitfieldExtract(index, 1, 2);
	y |= bitfieldExtract(index, 3, 2) << 1;
	x |= bitfieldExtract(index, 5, 1) << 2;
	return ivec2(x, y);
}

ivec2 unswizzle8x8(uint index)
{
	return unswizzle8x8_2x2_quant(index);
}

ivec2 unswizzle4x8_2x2_quant(uint index)
{
	uint y = bitfieldExtract(index, 0, 1);
	uint x = bitfieldExtract(index, 1, 2);
	y |= bitfieldExtract(index, 3, 2) << 1;
	return ivec2(x, y);
}

ivec2 unswizzle4x32(uint index)
{
	uint y = bitfieldExtract(index, 0, 1);
	uint x = bitfieldExtract(index, 1, 2);
	y |= bitfieldExtract(index, 3, 4) << 1;
	return ivec2(x, y);
}

#endif