// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#ifndef PYROWAVE_METAL_H_
#define PYROWAVE_METAL_H_

// Metal encoder and decoder for PyroWave, targeting Apple Silicon (Apple7 GPU family
// and up) on macOS and iOS.
//
// Metal objects cross the ABI as void *. This keeps the header usable from plain C,
// Objective-C and C++ alike, and sidesteps ARC ownership ambiguity at the boundary:
//   - Objective-C / ARC: pass (__bridge void *)mtlObject
//   - Objective-C, no ARC, or plain C: pass the object pointer directly
// Unless stated otherwise, objects passed in are retained for as long as PyroWave
// needs them and released on destroy, so the caller may release its own reference.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// API and ABI is not considered stable until MAJOR version hits 1!

#define PYROWAVE_API_VERSION_MAJOR 0
#define PYROWAVE_API_VERSION_MINOR 4
#define PYROWAVE_API_VERSION_PATCH 0

#if !defined(PYROWAVE_PUBLIC_API)
#if defined(PYROWAVE_EXPORT_SYMBOLS)
#if defined(__GNUC__)
#define PYROWAVE_PUBLIC_API __attribute__((visibility("default")))
#else
#define PYROWAVE_PUBLIC_API
#endif
#else
#define PYROWAVE_PUBLIC_API
#endif
#endif

// Codes 0 and -1 through -4 carry the same meaning as in pyrowave.h.
typedef enum pyrowave_result
{
	PYROWAVE_SUCCESS = 0,
	PYROWAVE_ERROR_GENERIC = -1,
	PYROWAVE_ERROR_INVALID_ARGUMENT = -2,
	PYROWAVE_ERROR_OUT_OF_HOST_MEMORY = -3,
	PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY = -4,
	PYROWAVE_ERROR_UNSUPPORTED_DEVICE = -5,
	PYROWAVE_ERROR_SHADER_COMPILATION = -6,
	PYROWAVE_ERROR_CORRUPT_BITSTREAM = -7,
	PYROWAVE_ERROR_INT_MAX = 0x7fffffff
} pyrowave_result;

typedef enum pyrowave_chroma_subsampling
{
	PYROWAVE_CHROMA_SUBSAMPLING_420 = 0,
	PYROWAVE_CHROMA_SUBSAMPLING_444 = 1,
	PYROWAVE_CHROMA_SUBSAMPLING_INT_MAX = 0x7fffffff
} pyrowave_chroma_subsampling;

// Opaque Metal handles. See the note at the top of this header on bridging.
typedef void *pyrowave_mtl_device;         // id<MTLDevice>
typedef void *pyrowave_mtl_command_buffer; // id<MTLCommandBuffer>
typedef void *pyrowave_mtl_texture;        // id<MTLTexture>
typedef void *pyrowave_iosurface;          // IOSurfaceRef

typedef struct pyrowave_device_opaque *pyrowave_device;
typedef struct pyrowave_decoder_opaque *pyrowave_decoder;
typedef struct pyrowave_encoder_opaque *pyrowave_encoder;

typedef void (*pyrowave_message_cb)(void *userdata, const char *msg);

// Used to dynamically detect any API/ABI incompatibility. This entry point is stable.
PYROWAVE_PUBLIC_API void
pyrowave_get_api_version(uint32_t *major, uint32_t *minor, uint32_t *patch);

// Returns a human readable string for a result code. Never returns NULL.
PYROWAVE_PUBLIC_API const char *
pyrowave_result_to_string(pyrowave_result result);

// Device API.

PYROWAVE_PUBLIC_API pyrowave_result pyrowave_create_default_device(pyrowave_device *device);

typedef struct pyrowave_device_create_info
{
	// The id<MTLDevice> to decode on, or NULL to use the system default device.
	pyrowave_mtl_device mtl_device;

	// Optional message callback, or NULL to print to stderr.
	pyrowave_message_cb message_callback;
	void *message_userdata;
} pyrowave_device_create_info;

// Reports whether an id<MTLDevice> can run the PyroWave decoder. Requires a
// 32-wide SIMD group, threadgroups of at least 128 threads and the Apple7 family
// feature set. Intel and AMD Macs are not supported.
PYROWAVE_PUBLIC_API bool
pyrowave_device_is_supported(pyrowave_mtl_device mtl_device);

// Creates the shared device object. This compiles the decode pipelines, so it is
// comparatively expensive; create one and share it across decoders.
//
// The PYROWAVE_PRECISION environment variable selects the wavelet precision, as
// on the Vulkan side: 2 is FP32, 1 (the default) keeps FP32 lifting math but
// stores the wavelet pyramid as FP16, and 0 is FP16 throughout. 1 is both the
// fastest and within 1 LSB of the FP32 result; 0 is slower than 1 and less
// accurate, so it exists mainly for comparison.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_device_create(const pyrowave_device_create_info *info, pyrowave_device *device);

// All decoders created from this device must be destroyed first.
PYROWAVE_PUBLIC_API void
pyrowave_device_destroy(pyrowave_device device);

// Encoder API
typedef struct pyrowave_encoder_create_info
{
	pyrowave_device device;

	// Luma dimensions. For 420 subsampling both must be even.
	// Both must be in the range [1, 16384], as the bitstream encodes them in 14 bits.
	int width;
	int height;

	pyrowave_chroma_subsampling chroma;
} pyrowave_encoder_create_info;

typedef struct pyrowave_packet
{
	size_t offset;
	size_t size;
} pyrowave_packet;

typedef struct pyrowave_rate_control
{
	// Very basic, target bitstream for an image must not exceed this size.
	size_t maximum_bitstream_size;
} pyrowave_rate_control;

typedef enum pyrowave_cpu_buffer_format
{
	PYROWAVE_CPU_BUFFER_FORMAT_NV12 = 0,    // 2 planes. Y in 8bpp, then CbCr interleaved in 16bpp.
	PYROWAVE_CPU_BUFFER_FORMAT_YUV420P = 1, // 3 planes, half resolution chroma.
	PYROWAVE_CPU_BUFFER_FORMAT_YUV444P = 2, // 3 planes, full resolution chroma.
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

// IOSurface input for the encoder. Two layouts are accepted:
//   - one biplanar NV12 surface in planes[0], with planes[1] and planes[2] NULL
//   - three single plane R8 surfaces, one per YUV component
// The library wraps these in MTLTextures itself, so the caller does not have to
// match any particular pixel format or usage. For the NV12 case the chroma plane
// is bound twice with different channel swizzles rather than being deinterleaved,
// so neither layout costs a copy.
typedef struct pyrowave_gpu_input
{
	pyrowave_iosurface planes[3];
} pyrowave_gpu_input;

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_create(const pyrowave_encoder_create_info *info, pyrowave_encoder *encoder);

PYROWAVE_PUBLIC_API void
pyrowave_encoder_destroy(pyrowave_encoder encoder);

// Both encode entry points submit GPU work and return without blocking; the
// subsequent packet queries block until it has finished. Encoding again clobbers
// the previous frame's result. The bitstream carries a small sequence counter so
// the decoder can track frame ordering.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_encode_gpu_synchronous(pyrowave_encoder encoder,
                                        const pyrowave_gpu_input *input,
                                        const pyrowave_rate_control *rate_control);

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_encode_cpu_synchronous(pyrowave_encoder encoder,
                                        const pyrowave_cpu_buffer *input,
                                        const pyrowave_rate_control *rate_control);

// Only valid after a successful encode, and only for that frame. Reports how many
// packets the frame needs if each may carry at most packet_boundary bytes.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_compute_num_packets(pyrowave_encoder encoder, size_t packet_boundary,
                                     size_t *num_packets);

// `packets` must have room for at least the count reported above; the number
// actually written is at most that and is returned in out_packets.
//
// Note packet_boundary is not a hard cap. A coded 32x32 block is the smallest unit
// a packet can carry, so a block that is on its own larger than packet_boundary
// yields one oversized packet rather than being split. Raising the rate control
// target makes individual blocks larger, so a caller sizing packets to an MTU
// should check the sizes it gets back rather than assume they fit.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_encoder_packetize(pyrowave_encoder encoder, pyrowave_packet *packets,
                           size_t packet_boundary, size_t *out_packets,
                           void *bitstream, size_t size);

// Decoder API
typedef struct pyrowave_decoder_create_info
{
	pyrowave_device device;

	// Luma dimensions. For 420 subsampling both must be even.
	// Both must be in the range [1, 16384], as the bitstream encodes them in 14 bits.
	int width;
	int height;

	pyrowave_chroma_subsampling chroma;
} pyrowave_decoder_create_info;

PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder);

PYROWAVE_PUBLIC_API void
pyrowave_decoder_destroy(pyrowave_decoder decoder);

// Throws away all queued packets.
PYROWAVE_PUBLIC_API void
pyrowave_decoder_clear(pyrowave_decoder decoder);

// A frame is potentially split into multiple packets.
// If a packet is pushed for a frame that is deemed to arrive earlier, it is dropped.
// A packet pushed for a frame with a higher sequence clears the queued frame and starts a new one.
// Packets are pushed until pyrowave_decoder_decode_is_ready() says the frame is ready.
// Returns PYROWAVE_ERROR_CORRUPT_BITSTREAM if the data does not parse.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size);

// For error correction purposes it may be acceptable to decode a frame that dropped
// some packets. With allow_partial_frame, a frame is considered ready once more than
// half of its blocks have arrived.
PYROWAVE_PUBLIC_API bool
pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame);

// Y, Cb, Cr as three separate single-channel textures.
//
// Requirements on each texture:
//   - MTLTextureType2D, created with MTLTextureUsageShaderWrite
//   - a single-channel writable format (r8Unorm, r16Unorm and r32Float are the
//     expected choices; the shader writes normalized float)
//   - plane 0 is width x height; for 420 chroma, planes 1 and 2 are
//     (width / 2) x (height / 2), and for 444 they match plane 0
//
// NV12 output is not yet supported.
typedef struct pyrowave_gpu_buffers
{
	pyrowave_mtl_texture planes[3];
} pyrowave_gpu_buffers;

// Encodes the decode work into the supplied id<MTLCommandBuffer>. Nothing is
// committed; the caller owns submission, and may encode MTLSharedEvent waits or
// signals on the same command buffer to synchronize with other work.
//
// Decoding may be requested at any time, producing incomplete results if packets
// are missing; missing wavelet coefficients are treated as 0, which shows up as
// extra blurring. Use pyrowave_decoder_decode_is_ready() to tell whether the
// result is known to be complete.
//
// The plane textures are written by compute shaders. Their prior contents are not
// read, so no pre-decode synchronization on them is needed beyond ordinary Metal
// command buffer ordering. Work encoded later in the same command buffer observes
// the results without explicit barriers; across command buffers, the usual Metal
// ordering guarantees on a single MTLCommandQueue apply.
//
// PyroWave holds a reference to the plane textures only for the duration of this
// call; the command buffer retains them thereafter as Metal normally does.
PYROWAVE_PUBLIC_API pyrowave_result
pyrowave_decoder_decode_gpu_buffer(pyrowave_decoder decoder,
                                   pyrowave_mtl_command_buffer command_buffer,
                                   const pyrowave_gpu_buffers *buffers);

#ifdef __cplusplus
}
#endif

#endif
