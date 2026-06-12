// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "context.hpp"
#include "device.hpp"
#include "image.hpp"
#include "buffer.hpp"
#include "pyrowave.h"
#include "pyrowave_decoder.hpp"
#include "pyrowave_encoder.hpp"

using namespace Granite;
using namespace Vulkan;
using namespace PyroWave;

extern "C" {
struct pyrowave_device_opaque
{
	Context context;
	Device device;
};

pyrowave_result pyrowave_create_default_device(pyrowave_device *device)
{
	if (!Context::init_loader(nullptr))
		return PYROWAVE_ERROR_NO_VULKAN;

	auto *dev = new pyrowave_device_opaque();
	dev->context.set_num_thread_indices(1);
	dev->context.set_system_handles({});
	if (!dev->context.init_instance_and_device(nullptr, 0, nullptr, 0))
		return PYROWAVE_ERROR_NO_VULKAN;

	dev->device.set_context(dev->context);
	*device = dev;
	return PYROWAVE_SUCCESS;
}

void pyrowave_device_destroy(pyrowave_device device)
{
	delete device;
}

struct pyrowave_encoder_opaque
{
	Device *device = nullptr;
	Encoder encoder;
	Fence queued_fence;
	BufferHandle queued_meta;
	BufferHandle queued_bitstream;
	ChromaSubsampling chroma = {};
	int width = 0;
	int height = 0;
};

pyrowave_result
pyrowave_encoder_create(const pyrowave_encoder_create_info *info, pyrowave_encoder *encoder)
{
	if (!info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->width <= 0 || info->height <= 0)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 && (info->width % 2 || info->height % 2))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *enc = new pyrowave_encoder_opaque();
	enc->device = &info->device->device;
	enc->chroma = ChromaSubsampling(info->chroma);
	enc->width = info->width;
	enc->height = info->height;

	if (!enc->encoder.init(&info->device->device, info->width, info->height, enc->chroma))
	{
		delete enc;
		return PYROWAVE_ERROR_GENERIC;
	}
	*encoder = enc;
	return PYROWAVE_SUCCESS;
}

struct WrappedViewBuffers : ViewBuffers
{
	ImageHandle wrapped_images[3];
	ImageViewHandle image_views[3];
	bool wrap(Device *device, const pyrowave_gpu_buffers *buffers, VkImageUsageFlags usage);
};

bool WrappedViewBuffers::wrap(Device *device, const pyrowave_gpu_buffers *buffers, VkImageUsageFlags usage)
{
	for (int i = 0; i < 3; i++)
	{
		ImageCreateInfo image_info = {};
		image_info.usage = usage;
		image_info.type = VK_IMAGE_TYPE_2D;
		image_info.domain = ImageDomain::Physical;
		image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
		image_info.width = buffers->planes[i].width;
		image_info.height = buffers->planes[i].height;
		image_info.format = buffers->planes[i].image_format;
		image_info.layers = buffers->planes[i].layer + 1;
		image_info.levels = buffers->planes[i].mip_level + 1;
		image_info.layout =
			buffers->planes[i].layout == VK_IMAGE_LAYOUT_GENERAL ? ImageLayout::General : ImageLayout::Optimal;
		wrapped_images[i] = device->wrap_image(image_info, buffers->planes[i].image);
		if (!wrapped_images[i])
			return false;

		ImageViewCreateInfo view_info = {};
		view_info.image = wrapped_images[i].get();
		view_info.format = buffers->planes[i].view_format;
		view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
		view_info.layers = 1;
		view_info.levels = 1;
		view_info.base_level = buffers->planes[i].mip_level;
		view_info.base_layer = buffers->planes[i].layer;
		view_info.swizzle.r = buffers->planes[i].swizzle;
		view_info.swizzle.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_info.swizzle.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_info.swizzle.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_info.aspect = buffers->planes[i].aspect;
		image_views[i] = device->create_image_view(view_info);
		if (!image_views[i])
			return false;

		planes[i] = image_views[i].get();
	}

	return true;
}

pyrowave_result
pyrowave_encoder_encode_gpu_synchronous(pyrowave_encoder encoder, const pyrowave_gpu_buffers *buffers,
										const pyrowave_rate_control *rate_control)
{
	auto *device = encoder->device;

	device->next_frame_context();

	BufferCreateInfo bufinfo = {};
	bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	bufinfo.size = encoder->encoder.get_meta_required_size();
	bufinfo.domain = BufferDomain::CachedHost;
	encoder->queued_meta = device->create_buffer(bufinfo);
	bufinfo.domain = BufferDomain::Device;
	auto queued_meta_gpu = device->create_buffer(bufinfo);

	auto target_bitstream_size = rate_control->maximum_bitstream_size & ~VkDeviceSize(3u);
	bufinfo.size = target_bitstream_size + encoder->encoder.get_meta_required_size();
	bufinfo.domain = BufferDomain::CachedHost;
	encoder->queued_bitstream = device->create_buffer(bufinfo);
	bufinfo.domain = BufferDomain::Device;
	auto queued_bitstream_gpu = device->create_buffer(bufinfo);

	Encoder::BitstreamBuffers bitstream_buffers = {};

	WrappedViewBuffers views = {};
	if (!views.wrap(device, buffers, VK_IMAGE_USAGE_SAMPLED_BIT))
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	bitstream_buffers.meta.buffer = queued_meta_gpu.get();
	bitstream_buffers.meta.size = queued_meta_gpu->get_create_info().size;
	bitstream_buffers.bitstream.buffer = queued_bitstream_gpu.get();
	bitstream_buffers.bitstream.size = queued_bitstream_gpu->get_create_info().size;
	bitstream_buffers.target_size = target_bitstream_size;

	auto cmd = device->request_command_buffer(CommandBuffer::Type::AsyncCompute);
	auto ret = encoder->encoder.encode(*cmd, views, bitstream_buffers);
	if (!ret)
	{
		device->submit_discard(cmd);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}
	cmd->copy_buffer(*encoder->queued_meta, *queued_meta_gpu);
	cmd->copy_buffer(*encoder->queued_bitstream, *queued_bitstream_gpu);
	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
				 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	encoder->queued_fence.reset();
	device->submit(cmd, &encoder->queued_fence);
	return PYROWAVE_SUCCESS;
}

pyrowave_result
pyrowave_encoder_encode_cpu_synchronous(pyrowave_encoder encoder, const pyrowave_cpu_buffer *buffers,
										const pyrowave_rate_control *rate_control)
{
	int num_planes = buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_NV12 ? 2 : 3;
	auto *device = encoder->device;
	ImageHandle images[3];

	if (buffers->width != encoder->width || buffers->height != encoder->height)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (encoder->chroma == ChromaSubsampling::Chroma420 && buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (encoder->chroma == ChromaSubsampling::Chroma444 && buffers->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	bool rdoc_capture = Device::init_renderdoc_capture();
	if (rdoc_capture)
		device->begin_renderdoc_capture();

	for (int plane = 0; plane < num_planes; plane++)
	{
		int plane_width = encoder->width;
		int plane_height = encoder->height;

		if (plane != 0 && encoder->chroma == ChromaSubsampling::Chroma420)
		{
			plane_width /= 2;
			plane_height /= 2;
		}

		const size_t plane_bpp = num_planes == 2 && plane == 1 ? 2 : 1;

		if (buffers->row_stride_in_bytes[plane] < plane_width * plane_bpp)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		if (buffers->row_stride_in_bytes[plane] * plane_height > buffers->plane_size_in_bytes[plane])
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	for (int plane = 0; plane < num_planes; plane++)
	{
		unsigned plane_bpp = num_planes == 2 && plane == 1 ? 2 : 1;

		const ImageInitialData initial = {
			buffers->data[plane],
			uint32_t(buffers->row_stride_in_bytes[plane] / plane_bpp)
		};

		auto info = ImageCreateInfo::immutable_2d_image(
			buffers->width, buffers->height,
			plane_bpp == 2 ? VK_FORMAT_R8G8_UNORM : VK_FORMAT_R8_UNORM);

		if (plane != 0 && encoder->chroma == ChromaSubsampling::Chroma420)
		{
			info.width /= 2;
			info.height /= 2;
		}

		images[plane] = device->create_image(info, &initial);
		if (!images[plane])
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	pyrowave_gpu_buffers gpu_buffers = {};

	for (int plane = 0; plane < 3; plane++)
	{
		auto &p = gpu_buffers.planes[plane];
		p.width = images[plane] ? images[plane]->get_width() : images[1]->get_width();
		p.height = images[plane] ? images[plane]->get_height() : images[1]->get_height();

		p.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		p.swizzle = num_planes == 2 && plane == 2 ? VK_COMPONENT_SWIZZLE_G : VK_COMPONENT_SWIZZLE_R;
		p.image_format = images[plane] ? images[plane]->get_format() : VK_FORMAT_R8G8_UNORM;
		p.view_format = p.image_format;
		p.layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
		p.image = images[plane] ? images[plane]->get_image() : images[1]->get_image();
	}

	auto ret = pyrowave_encoder_encode_gpu_synchronous(encoder, &gpu_buffers, rate_control);
	if (rdoc_capture)
		device->end_renderdoc_capture();
	return ret;
}

pyrowave_result
pyrowave_encoder_compute_num_packets(pyrowave_encoder encoder, size_t packet_boundary, size_t *num_packets)
{
	if (!encoder->queued_fence)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	encoder->queued_fence->wait();

	auto *mapped_meta = encoder->device->map_host_buffer(*encoder->queued_meta, MEMORY_ACCESS_READ_BIT);
	*num_packets = encoder->encoder.compute_num_packets(mapped_meta, packet_boundary);
	return PYROWAVE_SUCCESS;
}

pyrowave_result
pyrowave_encoder_packetize(pyrowave_encoder encoder, pyrowave_packet *packets, size_t packet_boundary,
                           size_t *out_packets, void *bitstream, size_t size)
{
	if (!encoder->queued_fence)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	encoder->queued_fence->wait();

	auto *mapped_meta = encoder->device->map_host_buffer(*encoder->queued_meta, MEMORY_ACCESS_READ_BIT);
	auto *mapped_bitstream = encoder->device->map_host_buffer(*encoder->queued_bitstream, MEMORY_ACCESS_READ_BIT);

	*out_packets = encoder->encoder.packetize(
		reinterpret_cast<Encoder::Packet *>(packets), packet_boundary, bitstream,
		size, mapped_meta, mapped_bitstream);

	return PYROWAVE_SUCCESS;
}

void pyrowave_encoder_destroy(pyrowave_encoder encoder)
{
	delete encoder;
}

struct pyrowave_decoder_opaque
{
	Device *device = nullptr;
	Decoder decoder;
	ImageHandle planes[3];
	bool fragment_path = false;
	ChromaSubsampling chroma = {};
	int width = 0;
	int height = 0;
};

bool pyrowave_decoder_device_prefers_fragment_path(pyrowave_device device)
{
	return Decoder::device_prefers_fragment_path(device->device);
}

pyrowave_result
pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder)
{
	if (!info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->width <= 0 || info->height <= 0)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 && (info->width % 2 || info->height % 2))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *dec = new pyrowave_decoder_opaque();
	dec->device = &info->device->device;
	dec->chroma = ChromaSubsampling(info->chroma);
	dec->fragment_path = info->fragment_path;
	dec->width = info->width;
	dec->height = info->height;

	if (!dec->decoder.init(dec->device, info->width, info->height, dec->chroma, info->fragment_path))
	{
		delete dec;
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	*decoder = dec;
	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_clear(pyrowave_decoder decoder)
{
	decoder->decoder.clear();
}

// A frame is potentially split into multiple packets.
pyrowave_result
pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size)
{
	bool ret = decoder->decoder.push_packet(data, size);
	return ret ? PYROWAVE_SUCCESS : PYROWAVE_ERROR_INVALID_ARGUMENT;
}

// For error correction purposes, it may be okay to decode a frame which dropped some packets.
bool pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame)
{
	return decoder->decoder.decode_is_ready(allow_partial_frame);
}

pyrowave_result
pyrowave_decoder_decode_gpu_buffer(pyrowave_decoder decoder, const pyrowave_gpu_buffers *buffers)
{
	auto *device = decoder->device;
	device->next_frame_context();

	WrappedViewBuffers views = {};
	if (!views.wrap(device, buffers, decoder->fragment_path ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_STORAGE_BIT))
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	// Just use normal graphics queue here since the result will likely be consumed there.
	auto cmd = device->request_command_buffer();

	auto ret = decoder->decoder.decode(*cmd, views);
	if (!ret)
	{
		device->submit_discard(cmd);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	device->submit(cmd);
	return PYROWAVE_SUCCESS;
}

pyrowave_result
pyrowave_decoder_decode_cpu_buffer_synchronous(pyrowave_decoder decoder, const pyrowave_cpu_buffer *buffers)
{
	auto *device = decoder->device;

	if (buffers->width != decoder->width || buffers->height != decoder->height)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_NV12)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (decoder->chroma == ChromaSubsampling::Chroma420 && buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (decoder->chroma == ChromaSubsampling::Chroma444 && buffers->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	bool rdoc_capture = Device::init_renderdoc_capture();
	if (rdoc_capture)
		device->begin_renderdoc_capture();

	for (int plane = 0; plane < 3; plane++)
	{
		int plane_width = decoder->width;
		int plane_height = decoder->height;

		if (plane != 0 && decoder->chroma == ChromaSubsampling::Chroma420)
		{
			plane_width /= 2;
			plane_height /= 2;
		}

		const size_t plane_bpp = 1;

		if (buffers->row_stride_in_bytes[plane] < plane_width * plane_bpp)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		if (buffers->row_stride_in_bytes[plane] * plane_height > buffers->plane_size_in_bytes[plane])
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	for (int plane = 0; plane < 3; plane++)
	{
		auto &img = decoder->planes[plane];

		if (!img)
		{
			auto info = ImageCreateInfo::immutable_2d_image(buffers->width, buffers->height, VK_FORMAT_R8_UNORM);
			info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			if (decoder->fragment_path)
			{
				info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			}
			else
			{
				info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
				info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
				info.layout = ImageLayout::General;
			}

			if (plane != 0 && decoder->chroma == ChromaSubsampling::Chroma420)
			{
				info.width /= 2;
				info.height /= 2;
			}

			img = device->create_image(info);
			if (!img)
				return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
		}
	}

	if (decoder->fragment_path)
	{
		auto cmd = device->request_command_buffer();
		cmd->begin_barrier_batch();
		for (auto &img : decoder->planes)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
				VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
		}
		cmd->end_barrier_batch();
		device->submit(cmd);
	}

	pyrowave_gpu_buffers gpu_buffers = {};
	for (int plane = 0; plane < 3; plane++)
	{
		auto &p = gpu_buffers.planes[plane];
		p.image = decoder->planes[plane]->get_image();
		p.width = decoder->planes[plane]->get_width();
		p.height = decoder->planes[plane]->get_height();
		p.image_format = decoder->planes[plane]->get_format();
		p.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		p.swizzle = VK_COMPONENT_SWIZZLE_IDENTITY;
		p.view_format = p.image_format;
		p.layout = decoder->fragment_path ? VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	}

	BufferCreateInfo bufinfo = {};
	BufferHandle readback_buffers[3];

	auto res = pyrowave_decoder_decode_gpu_buffer(decoder, &gpu_buffers);
	if (res != PYROWAVE_SUCCESS)
		return res;

	auto cmd = device->request_command_buffer();

	if (decoder->fragment_path)
	{
		cmd->begin_barrier_batch();
		for (auto &img : decoder->planes)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
		}
		cmd->end_barrier_batch();
	}
	else
	{
		cmd->barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
	}

	for (int plane = 0; plane < 3; plane++)
	{
		bufinfo.size = buffers->plane_size_in_bytes[plane];
		bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufinfo.domain = BufferDomain::CachedHost;
		readback_buffers[plane] = device->create_buffer(bufinfo);

		cmd->copy_image_to_buffer(*readback_buffers[plane], *decoder->planes[plane], 0, {},
		                          {decoder->planes[plane]->get_width(), decoder->planes[plane]->get_height(), 1},
		                          buffers->row_stride_in_bytes[plane], 0,
		                          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
	}

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

	Fence fence;
	device->submit(cmd, &fence);
	fence->wait();

	for (int plane = 0; plane < 3; plane++)
	{
		void *mapped = device->map_host_buffer(*readback_buffers[plane], MEMORY_ACCESS_READ_BIT);
		memcpy(buffers->data[plane], mapped, buffers->plane_size_in_bytes[plane]);
	}

	if (rdoc_capture)
		device->end_renderdoc_capture();

	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_destroy(pyrowave_decoder decoder)
{
	delete decoder;
}
}