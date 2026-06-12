// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#pragma once

// C99 header

#if !defined(VULKAN_CORE_H_)
#error "Must include vulkan headers before including pyrowave.h"
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#define PYROWAVE_API_VERSION 1

#if !defined(PYROWAVE_PUBLIC_API)
#if defined(PYROWAVE_EXPORT_SYMBOLS)
#if defined(__GNUC__)
#define PYROWAVE_PUBLIC_API __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define PYROWAVE_PUBLIC_API __declspec(dllexport)
#else
#define PYROWAVE_PUBLIC_API
#endif
#else
#define PYROWAVE_PUBLIC_API
#endif
#else
#define PYROWAVE_PUBLIC_API
#endif

typedef enum pyrowave_result
{
	PYROWAVE_SUCCESS = 0,
	PYROWAVE_ERROR_GENERIC = -1,
	PYROWAVE_ERROR_INVALID_ARGUMENT = -2,
	PYROWAVE_ERROR_OUT_OF_HOST_MEMORY = -3,
	PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY = -4,
	PYROWAVE_ERROR_NO_VULKAN = -5,
	PYROWAVE_ERROR_INT_MAX = 0x7fffffff
} pyrowave_result;

typedef enum pyrowave_chroma_subsampling
{
	PYROWAVE_CHROMA_SUBSAMPLING_420 = 0,
	PYROWAVE_CHROMA_SUBSAMPLING_444 = 1,
	PYROWAVE_CHROMA_SUBSAMPLING_INT_MAX = 0x7fffffff
} pyrowave_chroma_subsampling;

typedef struct pyrowave_encoder_opaque *pyrowave_encoder;
typedef struct pyrowave_decoder_opaque *pyrowave_decoder;
typedef struct pyrowave_device_opaque *pyrowave_device;

// Device API.
PYROWAVE_PUBLIC_API pyrowave_result pyrowave_create_default_device(pyrowave_device *device);

// TODO: Add interop where device can be created from existing VkInstance/VkPhysicalDevice/VkDevice.
// TODO: Add interop where device can be created from LUID (Windows).
// TODO: Add interop where device can be created from some other compatibility information (general).

// All encoders and decoders must have been destroyed before destroying the device.
PYROWAVE_PUBLIC_API void pyrowave_device_destroy(pyrowave_device device);
////

// Encoder API
typedef struct pyrowave_encoder_create_info
{
	pyrowave_device device;
	// For 420 subsampling, must be even.
	int width;
	int height;
	pyrowave_chroma_subsampling chroma;
} pyrowave_encoder_create_info;

typedef struct pyrowave_packet
{
	size_t offset;
	size_t size;
} pyrowave_packet;

typedef struct pyrowave_image_view
{
	VkImage image;
	// Extent of mip0. Must be consistent with width/height used to create the encoder.
	uint32_t width;
	uint32_t height;
	// Base format used to create the image.
	VkFormat image_format;

	// Must be UNORM in some way and be supported for sampling and storage.
	// For planar image_format, the image must have been created with
	// VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT to be able to take plane views.
	VkFormat view_format;
	uint32_t mip_level;
	uint32_t layer;
	// If using a planar image format, needs to be e.g. VK_IMAGE_ASPECT_PLANE_*_BIT.
	VkImageAspectFlagBits aspect;
	// For decode path, must be IDENTITY or R.
	VkComponentSwizzle swizzle;
	// Must be VK_IMAGE_LAYOUT_(SHADER_)READ_ONLY_OPTIMAL (encode only) or VK_IMAGE_LAYOUT_GENERAL.
	// For fragment decode path, must be (COLOR_)ATTACHMENT_OPTIMAL or VK_IMAGE_LAYOUT_GENERAL.
	// Pyrowave will not perform any image layout transitions on its own in the GPU buffer paths.
	VkImageLayout layout;
} pyrowave_image_view;

typedef struct pyrowave_gpu_buffers
{
	// All 3 planes must be provided. For NV12 images, pass in the same plane for Cb and Cr, but use swizzle
	// to select R and G planes to fake a YUV420P image.
	// Very slightly less efficient, but should barely be measurable.
	pyrowave_image_view planes[3];

	// TODO: Add synchronization information.
} pyrowave_gpu_buffers;

// TODO: Add support for importing external memory as GPU buffers.

// The CPU path is mostly for bringup testing.
typedef enum pyrowave_cpu_buffer_format
{
	PYROWAVE_CPU_BUFFER_FORMAT_NV12 = 0, // 2 planes. Y packed in 8bpp, then CbCr packed in 16bpp. Only supported for encoding.
	PYROWAVE_CPU_BUFFER_FORMAT_YUV420P = 1, // 3 planes. Y, Cb, Cr packed into separate planes. Native format for pyrowave.
	PYROWAVE_CPU_BUFFER_FORMAT_YUV444P = 2, // 3 planes. Y, Cb, Cr packed into separate planes. Native format for pyrowave.
	PYROWAVE_CPU_BUFFER_FORMAT_INT_MAX = 0x7fffffff
} pyrowave_cpu_buffer_format;

typedef struct pyrowave_cpu_buffer
{
	// Written in decoder, read-only in encoder.
	void *data[3];
	// Must be at least width for plane times texel size of the plane.
	size_t row_stride_in_bytes[3];
	// Must be at least row_stride times height of plane.
	size_t plane_size_in_bytes[3];
	// Size of the luma plane. Size of chroma is implied by format.
	// Must be same extent as decoder.
	int width;
	int height;
	pyrowave_cpu_buffer_format format;
} pyrowave_cpu_buffer;

typedef struct pyrowave_rate_control
{
	// Very basic, target bitstream for an image must not exceed this size.
	size_t maximum_bitstream_size;
} pyrowave_rate_control;

// The entry points for encoder are not thread safe. Application must ensure synchronization.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_create(const pyrowave_encoder_create_info *info, pyrowave_encoder *encoder);

// Synchronous encode API. For low-latency use cases, overlapping frames in encode is meaningless
// due to latency and the encoder is so fast anyway. This function will not block, but subsequent functions will.
// Calling an encode operation with synchronous API clobbers any previous encoded frame.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_encode_gpu_synchronous(pyrowave_encoder encoder, const pyrowave_gpu_buffers *buffers,
                                        const pyrowave_rate_control *rate_control);

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_encode_cpu_synchronous(pyrowave_encoder encoder, const pyrowave_cpu_buffer *buffers,
                                        const pyrowave_rate_control *rate_control);

// Can only be called after a successful encoding operation and result is only valid for that particular frame.
// Computes the number of network packets required if each packet can consume a provided number of bytes.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_compute_num_packets(pyrowave_encoder encoder, size_t packet_boundary, size_t *num_packets);

// Number of packets is implied to be greater-than-equal to num_packets as returned earlier.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_packetize(pyrowave_encoder encoder, pyrowave_packet *packets, size_t packet_boundary,
                           size_t *out_packets, void *bitstream, size_t size);

// Implementation ensures GPU is idle before destroying objects.
PYROWAVE_PUBLIC_API void
pyrowave_encoder_destroy(pyrowave_encoder encoder);
//////

// Decoder
typedef struct pyrowave_decoder_create_info
{
	pyrowave_device device;
	// For 420 subsampling, must be even.
	int width;
	int height;
	pyrowave_chroma_subsampling chroma;
	bool fragment_path;
} pyrowave_decoder_create_info;

// Fragment path is optimized for typical mobile GPUs which have weak compute support.
// iDWT is instead computed entirely in traditional render passes and fragment shaders.
// This path is *not* recommended for desktop-class chips.
PYROWAVE_PUBLIC_API bool
pyrowave_decoder_device_prefers_fragment_path(pyrowave_device device);

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder);

// Throws away all queued packets.
PYROWAVE_PUBLIC_API void pyrowave_decoder_clear(pyrowave_decoder decoder);

// A frame is potentially split into multiple packets.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size);

// For error correction purposes, it may be okay to decode a frame which dropped some packets.
PYROWAVE_PUBLIC_API bool
pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame);

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_decode_gpu_buffer(pyrowave_decoder decoder, const pyrowave_gpu_buffers *buffers);

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_decode_cpu_buffer_synchronous(pyrowave_decoder decoder, const pyrowave_cpu_buffer *buffers);

// Implementation ensures GPU is idle before destroying objects.
PYROWAVE_PUBLIC_API void pyrowave_decoder_destroy(pyrowave_decoder decoder);
//////

#ifdef __cplusplus
}
#endif
