// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#include "pyrowave_common.hpp"

#if PYROWAVE_PRECISION < 0 || PYROWAVE_PRECISION > 2
#error "PYROWAVE_PRECISION must be in range [0, 2]."
#endif

constexpr int WaveletFP16Levels = 2;

namespace PyroWave
{
using namespace Vulkan;

Configuration::Configuration()
{
	precision = PYROWAVE_PRECISION;
	if (const char *env = getenv("PYROWAVE_PRECISION"))
		precision = int(strtol(env, nullptr, 0));

	if (precision < 0 || precision > 2)
	{
		fprintf(stderr, "pyrowave: precision must be in range [0, 2].\n");
		precision = PYROWAVE_PRECISION;
	}

	LOGI("Selection precision level: %d\n", precision);
}

Configuration &Configuration::get()
{
	static Configuration config;
	return config;
}

int Configuration::get_precision() const
{
	return precision;
}

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
			Configuration::get().get_precision() == 2 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R16_SFLOAT);
	info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
	             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	info.layers = NumFrequencyBandsPerLevel * NumComponents;
	info.levels = Configuration::get().get_precision() != 1 ? DecompositionLevels : WaveletFP16Levels;

	wavelet_img_high_res = device->create_image(info);
	wavelet_img_high_res->set_layout(Layout::General);
	device->set_name(*wavelet_img_high_res, "wavelet-buffer-high-res");

	if (Configuration::get().get_precision() == 1)
	{
		// For the lowest level bands, we want to maintain precision as much as possible and bandwidth here is trivial.
		info.levels = DecompositionLevels - info.levels;
		info.format = VK_FORMAT_R32_SFLOAT;
		info.width >>= WaveletFP16Levels;
		info.height >>= WaveletFP16Levels;
		wavelet_img_low_res = device->create_image(info);
		wavelet_img_low_res->set_layout(Layout::General);
		device->set_name(*wavelet_img_low_res, "wavelet-buffer-low-res");
	}

	for (int level = 0; level < DecompositionLevels; level++)
	{
		ImageViewCreateInfo view_info = {};
		view_info.levels = 1;
		view_info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;

		if (Configuration::get().get_precision() != 1 || level < WaveletFP16Levels)
		{
			view_info.base_level = level;
			view_info.image = wavelet_img_high_res.get();
		}
		else
		{
			view_info.base_level = level - WaveletFP16Levels;
			view_info.image = wavelet_img_low_res.get();
		}

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
			if (level == 0 && component != 0 && chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				uint32_t level_width = wavelet_img_high_res->get_width(level);
				uint32_t level_height = wavelet_img_high_res->get_height(level);

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

bool WaveletBuffers::init(Device *device_, int width_, int height_, ChromaSubsampling chroma_)
{
	device = device_;
	width = width_;
	height = height_;
	chroma = chroma_;

	aligned_width = align(width, Alignment);
	aligned_height = align(height, Alignment);
	aligned_width = std::max<int>(aligned_width, MinimumImageSize);
	aligned_height = std::max<int>(aligned_height, MinimumImageSize);

	init_samplers();
	allocate_images();

	init_block_meta();

	Vulkan::ResourceLayout layout;

	shaders = Shaders<>(*device, layout, [this](const char *, const char *env) {
		if (strcmp(env, "FP16") == 0)
			return device->get_device_features().vk12_features.shaderFloat16 ? 1 : 0;
		return 0;
	});

	return true;
}
}
