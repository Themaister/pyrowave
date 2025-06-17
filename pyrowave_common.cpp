// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#include "pyrowave_common.hpp"

namespace PyroWave
{
using namespace Vulkan;

void WaveletBuffers::init_samplers()
{
	SamplerCreateInfo samp = {};
	samp.address_mode_u = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
	samp.address_mode_v = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
	samp.address_mode_w = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
	mirror_repeat_sampler = device->create_sampler(samp);

	samp.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samp.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samp.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samp.border_color = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	border_sampler = device->create_sampler(samp);
}

void WaveletBuffers::allocate_images()
{
	auto info = ImageCreateInfo::immutable_2d_image(
			aligned_width / 2, aligned_height / 2,
			HighPrecision ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R16_SFLOAT);
	info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
	             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	info.layers = NumFrequencyBandsPerLevel * NumComponents;
	info.levels = DecompositionLevels;

	wavelet_img = device->create_image(info);
	wavelet_img->set_layout(Layout::General);
	device->set_name(*wavelet_img, "wavelet-buffer");

	for (int level = 0; level < DecompositionLevels; level++)
	{
		ImageViewCreateInfo view_info = {};
		view_info.levels = 1;
		view_info.base_level = level;
		view_info.image = wavelet_img.get();
		view_info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;

		for (int component = 0; component < NumComponents; component++)
		{
			view_info.base_layer = 4 * component;

			view_info.view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
			view_info.layers = 4;
			component_layer_views[component][level] = device->create_image_view(view_info);

			view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
			view_info.layers = 1;
			component_ll_views[component][level] = device->create_image_view(view_info);
		}
	}
}

void WaveletBuffers::accumulate_block_mapping(int blocks_x_8x8, int blocks_y_8x8)
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

void WaveletBuffers::init_block_meta()
{
	for (int level = DecompositionLevels - 1; level >= 0; level--)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				uint32_t level_width = wavelet_img->get_width(level);
				uint32_t level_height = wavelet_img->get_height(level);

				int blocks_x_8x8 = (level_width + 7) / 8;
				int blocks_y_8x8 = (level_height + 7) / 8;
				int blocks_x_32x32 = (level_width + 31) / 32;

				block_meta[component][level][band] = {
					block_count_8x8, blocks_x_8x8,
					block_count_32x32, blocks_x_32x32,
				};

				accumulate_block_mapping(blocks_x_8x8, blocks_y_8x8);
			}
		}
	}
}

bool WaveletBuffers::init(Device *device_, int width_, int height_)
{
	device = device_;
	width = width_;
	height = height_;

	aligned_width = align(width, Alignment);
	aligned_height = align(height, Alignment);
	aligned_width = std::max<int>(aligned_width, MinimumImageSize);
	aligned_height = std::max<int>(aligned_height, MinimumImageSize);

	init_samplers();
	allocate_images();

	init_block_meta();

	Vulkan::ResourceLayout layout;
	shaders = Shaders<>(*device, layout, 0);

	return true;
}
}
