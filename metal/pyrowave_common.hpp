// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

// Internal glue shared by the Metal encoder and decoder: the object behind
// pyrowave_device, the library/pipeline helpers, and the wavelet coefficient
// pyramid, which both directions allocate identically.
//
// This is not a public header, and it is Objective-C++: the backends are .mm so they
// can use Metal directly while keeping the C++ bitstream layer. Everything here is
// under ARC, so the id<MTL...> members below are strong references and nothing needs
// explicit retain or release.

#import <Metal/Metal.h>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"

#include <mutex>
#include <stdint.h>

namespace PyroWave
{
// Threadgroup sizes are baked into the shaders as local_size_x, except for
// resolve_rate_control, which declares it as a specialization constant.
constexpr uint32_t DequantThreadgroupSize = 128;
constexpr uint32_t IdwtThreadgroupSize = 64;
constexpr uint32_t DwtThreadgroupSize = 64;
constexpr uint32_t QuantThreadgroupSize = 128;
constexpr uint32_t AnalyzeThreadgroupSize = 64;
constexpr uint32_t AnalyzeFinalizeThreadgroupSize = 512;
constexpr uint32_t BlockPackingThreadgroupSize = 64;
// resolve_rate_control needs exactly one SIMD group per threadgroup, and every
// Apple GPU we support is 32 wide.
constexpr uint32_t ResolveThreadgroupSize = 32;

// Same meaning and same env var as the Vulkan build:
//   2  FP32 storage and math
//   1  (default) FP16 storage, FP32 lifting math, half sized threadgroup array
//   0  FP16 throughout
//
// 1 is the default because it is both the fastest and essentially free: on an
// M1 it decodes 1080p roughly 1.4x quicker than FP32 while staying within 1 LSB
// of the Vulkan decoder. The win is the halved memory traffic through the R16F
// wavelet pyramid, which is what the decode is bound by above 480p -- 0 is
// actually slower than 1 despite the half precision arithmetic, and costs
// accuracy, so it is not recommended.
//
// Note 1 is not quite identical to the Vulkan build's: that one additionally
// splits the low frequency bands into a separate FP32 image, which is not
// implemented here.
constexpr int DefaultPrecision = 1;

int requested_precision();
MTLPixelFormat wavelet_format(int precision);

const char *result_string(pyrowave_result result);

id<MTLLibrary> compile_library(pyrowave_device device, const char *source, const char *label);

// `constants` may be nil when the shader has no specialization constants to set.
id<MTLComputePipelineState> create_pipeline(pyrowave_device device, id<MTLLibrary> library,
                                           const char *entry_point, uint32_t required_threads,
                                           MTLFunctionConstantValues *constants = nil);

// Convenience for a single bool function constant. Note Metal refuses to build a
// pipeline from an unspecialized function if the shader declares any function
// constants at all, even ones SPIRV-Cross guarded with
// is_function_constant_defined(), so constants that want their default value
// still have to be set explicitly.
id<MTLComputePipelineState> create_pipeline_bool_constant(pyrowave_device device, id<MTLLibrary> library,
                                                          const char *entry_point,
                                                          uint32_t required_threads,
                                                          uint32_t index, bool value);

// The wavelet coefficient pyramid. The decoder fills it from the bitstream and
// runs the iDWT out of it; the encoder runs the DWT into it and quantizes out of
// it. Both want exactly the same texture and the same set of views, so this is
// shared verbatim.
struct WaveletPyramid
{
	// 2D array, NumFrequencyBandsPerLevel * NumComponents layers, DecompositionLevels mips.
	id<MTLTexture> texture;
	// 4 layer array views, one per component and level. Written as an image and
	// sampled as an array.
	id<MTLTexture> component_layer_views[NumComponents][DecompositionLevels];
	// Single layer 2D views of band 0 (LL).
	id<MTLTexture> component_ll_views[NumComponents][DecompositionLevels];

	bool init(pyrowave_device device, const BlockLayout &layout);

	WaveletPyramid() = default;
	WaveletPyramid(const WaveletPyramid &) = delete;
	void operator=(const WaveletPyramid &) = delete;
};
}

struct pyrowave_device_opaque
{
	id<MTLDevice> mtl;

	// Decode. Compiled by pyrowave_device_create().
	id<MTLComputePipelineState> dequant_pipeline;
	// Indexed by the DCShift function constant.
	id<MTLComputePipelineState> idwt_pipeline[2];

	// Encode. Compiled on demand by the first pyrowave_encoder_create(), so that
	// decode-only users do not pay for six extra shader compiles.
	id<MTLComputePipelineState> dwt_pipeline[2];
	id<MTLComputePipelineState> quant_pipeline;
	id<MTLComputePipelineState> analyze_pipeline;
	id<MTLComputePipelineState> analyze_finalize_pipeline;
	id<MTLComputePipelineState> resolve_pipeline;
	id<MTLComputePipelineState> block_packing_pipeline;
	bool encode_pipelines_ready = false;
	std::mutex encode_pipeline_lock;

	id<MTLSamplerState> mirror_repeat_sampler;
	// Clamp to transparent black. Only the encoder's quantizer wants this, so it
	// is created together with the encode pipelines.
	id<MTLSamplerState> border_sampler;

	pyrowave_message_cb message_cb = nullptr;
	void *message_userdata = nullptr;
	int precision = PyroWave::DefaultPrecision;

	void log(const char *fmt, ...) const __attribute__((format(printf, 2, 3)));
};
