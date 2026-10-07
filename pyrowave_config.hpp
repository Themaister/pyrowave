// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

namespace Vulkan
{
class ImageView;
}

namespace PyroWave
{
struct ViewBuffers
{
	const Vulkan::ImageView *planes[3] = {};
	float range_scale = 1.0f; // For encode. Used to scale e.g. yuv420p10 into full unorm range. Decoder, the inverse.
};

enum class ChromaSubsampling
{
	Chroma420,
	Chroma444
};
}
