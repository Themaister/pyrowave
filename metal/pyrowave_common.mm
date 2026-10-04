// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Device object, shader helpers and wavelet pyramid shared by the Metal encoder
// and decoder. Objective-C++ under ARC.

#include "pyrowave_common.hpp"

#include "shaders/pyrowave_msl.h"

#ifdef PYROWAVE_METAL_BENCH_HOOKS
#include "experimental_dequant.msl.h"
#include "experimental_idwt.msl.h"
#include "experimental_native_dequant.msl.h"
#include "experimental_native_idwt.msl.h"
#include "experimental_fused_idwt.msl.h"
#include "experimental_render.msl.h"
#endif

#include <memory>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

namespace PyroWave
{
int requested_precision()
{
	const char *env = getenv("PYROWAVE_PRECISION");
	if (!env)
		return DefaultPrecision;
	int precision = atoi(env);
	return (precision < 0 || precision > 2) ? DefaultPrecision : precision;
}

MTLPixelFormat wavelet_format(int precision)
{
	return precision == 2 ? MTLPixelFormatR32Float : MTLPixelFormatR16Float;
}

const char *result_string(pyrowave_result result)
{
	switch (result)
	{
	case PYROWAVE_SUCCESS: return "success";
	case PYROWAVE_ERROR_GENERIC: return "generic error";
	case PYROWAVE_ERROR_INVALID_ARGUMENT: return "invalid argument";
	case PYROWAVE_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
	case PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
	case PYROWAVE_ERROR_UNSUPPORTED_DEVICE: return "unsupported device";
	case PYROWAVE_ERROR_SHADER_COMPILATION: return "shader compilation failed";
	case PYROWAVE_ERROR_CORRUPT_BITSTREAM: return "corrupt bitstream";
	default: return "unknown error";
	}
}

id<MTLLibrary> compile_library(pyrowave_device device, const char *source, const char *label)
{
	NSError *error = nil;
	// Default (fast) math is kept deliberately. MathModeSafe was measured and does
	// not reduce the residual ~1 LSB disagreement with the Vulkan decoder, so the
	// difference does not come from FMA contraction or reassociation.
	id<MTLLibrary> library = [device->mtl newLibraryWithSource:@(source)
	                                                  options:nil
	                                                    error:&error];
	if (!library)
	{
		device->log("Failed to compile %s: %s", label,
		            error ? error.localizedDescription.UTF8String : "unknown error");
	}

	return library;
}

id<MTLComputePipelineState> create_pipeline(pyrowave_device device, id<MTLLibrary> library,
                                           const char *entry_point, uint32_t required_threads,
                                           MTLFunctionConstantValues *constants)
{
	NSError *error = nil;
	NSString *name = @(entry_point);

	id<MTLFunction> function;
	if (constants)
		function = [library newFunctionWithName:name constantValues:constants error:&error];
	else
		function = [library newFunctionWithName:name];

	if (!function)
	{
		device->log("Failed to look up %s: %s", entry_point,
		            error ? error.localizedDescription.UTF8String : "not found");
		return nil;
	}

	id<MTLComputePipelineState> pipeline =
			[device->mtl newComputePipelineStateWithFunction:function error:&error];

	if (!pipeline)
	{
		device->log("Failed to create pipeline for %s: %s", entry_point,
		            error ? error.localizedDescription.UTF8String : "unknown error");
		return nil;
	}

	if (pipeline.maxTotalThreadsPerThreadgroup < required_threads)
	{
		device->log("%s only supports %u threads per threadgroup, needs %u.",
		            entry_point, unsigned(pipeline.maxTotalThreadsPerThreadgroup), required_threads);
		return nil;
	}

	return pipeline;
}

id<MTLComputePipelineState> create_pipeline_bool_constant(pyrowave_device device, id<MTLLibrary> library,
                                                          const char *entry_point,
                                                          uint32_t required_threads,
                                                          uint32_t index, bool value)
{
	MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
	[constants setConstantValue:&value type:MTLDataTypeBool atIndex:index];
	return create_pipeline(device, library, entry_point, required_threads, constants);
}

#ifdef PYROWAVE_METAL_BENCH_HOOKS
pyrowave_result ensure_native_dequant_pipelines(pyrowave_device device, bool hybrid)
{
	std::lock_guard<std::mutex> holder{device->bench_native_pipeline_lock};
	if (device->bench_native_dequant_pipeline[hybrid] && device->bench_native_batched_dequant_pipeline[hybrid])
		return device->bench_native_dequant_pipeline[hybrid].threadExecutionWidth == 32 &&
		       device->bench_native_batched_dequant_pipeline[hybrid].threadExecutionWidth == 32 ?
		       PYROWAVE_SUCCESS : PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
	const auto &source = hybrid ? hybrid_dequant_source() : native_dequant_source();
	std::string batched;
	if (source.empty() || !build_batched_dequant_msl(source.c_str(), batched))
		return PYROWAVE_ERROR_SHADER_COMPILATION;
	auto *library = compile_library(device, source.c_str(), "native packed dequant");
	auto *batch_library = compile_library(device, batched.c_str(), "native packed batched dequant");
	if (!library || !batch_library) return PYROWAVE_ERROR_SHADER_COMPILATION;
	device->bench_native_dequant_pipeline[hybrid] = create_pipeline(device, library, "pyrowave_wavelet_dequant", DequantThreadgroupSize);
	device->bench_native_batched_dequant_pipeline[hybrid] = create_pipeline(device, batch_library, "pyrowave_wavelet_dequant_batched", DequantThreadgroupSize);
	return device->bench_native_dequant_pipeline[hybrid] && device->bench_native_batched_dequant_pipeline[hybrid] &&
	       device->bench_native_dequant_pipeline[hybrid].threadExecutionWidth == 32 &&
	       device->bench_native_batched_dequant_pipeline[hybrid].threadExecutionWidth == 32 ?
	       PYROWAVE_SUCCESS : PYROWAVE_ERROR_SHADER_COMPILATION;
}

pyrowave_result ensure_native_idwt_pipelines(pyrowave_device device)
{
	static_assert(IdwtThreadgroupSize == 64, "Native iDWT load mapping requires 64 threads.");
	std::lock_guard<std::mutex> holder{device->bench_native_pipeline_lock};
	if (device->bench_native_idwt_pipeline[0] && device->bench_native_idwt_pipeline[1])
		return PYROWAVE_SUCCESS;
	const auto &source = native_idwt_source(device->precision);
	if (source.empty()) return PYROWAVE_ERROR_SHADER_COMPILATION;
	auto *library = compile_library(device, source.c_str(), "native tiled idwt");
	if (!library) return PYROWAVE_ERROR_SHADER_COMPILATION;
	for (int i = 0; i < 2; i++)
		device->bench_native_idwt_pipeline[i] = create_pipeline_bool_constant(device, library, "pyrowave_idwt", IdwtThreadgroupSize, 0, i != 0);
	return device->bench_native_idwt_pipeline[0] && device->bench_native_idwt_pipeline[1] ?
	       PYROWAVE_SUCCESS : PYROWAVE_ERROR_SHADER_COMPILATION;
}

pyrowave_result ensure_fused_idwt_pipeline(pyrowave_device device, bool compact)
{
	static_assert(IdwtThreadgroupSize == 64, "Fused iDWT mapping requires 64 threads.");
	std::lock_guard<std::mutex> holder{device->bench_native_pipeline_lock};
	if (device->bench_fused_idwt_pipeline[compact]) return PYROWAVE_SUCCESS;
	const auto &source = compact ? compact_fused_idwt_source(device->precision) : phase_aligned_fused_idwt_source(device->precision);
	if (source.empty()) return PYROWAVE_ERROR_SHADER_COMPILATION;
	auto *library = compile_library(device, source.c_str(), "fused two-level idwt");
	if (!library) return PYROWAVE_ERROR_SHADER_COMPILATION;
	device->bench_fused_idwt_pipeline[compact] = create_pipeline_bool_constant(device, library, "pyrowave_idwt_fused", IdwtThreadgroupSize, 0, true);
	return device->bench_fused_idwt_pipeline[compact] ? PYROWAVE_SUCCESS : PYROWAVE_ERROR_SHADER_COMPILATION;
}

pyrowave_result ensure_rgb_idwt_pipeline(pyrowave_device device, bool chroma_420)
{
	static_assert(IdwtThreadgroupSize == 64, "Fused RGB mapping requires 64 threads.");
	std::lock_guard<std::mutex> holder{device->bench_native_pipeline_lock};
	if (device->bench_rgb_idwt_pipeline[chroma_420]) return PYROWAVE_SUCCESS;
	auto source = bench_fused_rgb_source(device->precision);
	if (source.empty()) return PYROWAVE_ERROR_SHADER_COMPILATION;
	auto *library = compile_library(device, source.c_str(), "fused idwt RGB");
	if (!library) return PYROWAVE_ERROR_SHADER_COMPILATION;
	device->bench_rgb_idwt_pipeline[chroma_420] = create_pipeline_bool_constant(device, library,
		chroma_420 ? "pyrowave_idwt_rgb420" : "pyrowave_idwt_rgb", IdwtThreadgroupSize, 0, true);
	return device->bench_rgb_idwt_pipeline[chroma_420] ? PYROWAVE_SUCCESS : PYROWAVE_ERROR_SHADER_COMPILATION;
}

pyrowave_result ensure_batched_dequant_pipeline(pyrowave_device device)
{
	std::lock_guard<std::mutex> holder{device->bench_batched_dequant_lock};
	if (device->bench_batched_dequant_pipeline)
		return PYROWAVE_SUCCESS;

	std::string source;
	try
	{
		if (!build_batched_dequant_msl(wavelet_dequant_msl_source, source))
		{
			device->log("Canonical dequant shader changed; refusing to build the benchmark variant.");
			return PYROWAVE_ERROR_SHADER_COMPILATION;
		}
	}
	catch (const std::bad_alloc &)
	{
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;
	}

	auto *library = compile_library(device, source.c_str(), "experimental batched wavelet_dequant");
	if (!library)
		return PYROWAVE_ERROR_SHADER_COMPILATION;
	auto *pipeline = create_pipeline(device, library, "pyrowave_wavelet_dequant_batched", DequantThreadgroupSize);
	if (!pipeline || pipeline.threadExecutionWidth != 32)
		return PYROWAVE_ERROR_SHADER_COMPILATION;
	device->bench_batched_dequant_pipeline = pipeline;
	return PYROWAVE_SUCCESS;
}

pyrowave_result ensure_reduced_barrier_idwt_pipelines(pyrowave_device device)
{
	static_assert(IdwtThreadgroupSize == 64, "The reduced iDWT barrier proof requires exactly 64 threads.");
	std::lock_guard<std::mutex> holder{device->bench_reduced_barrier_idwt_lock};
	if (device->bench_reduced_barrier_idwt_pipeline[0] && device->bench_reduced_barrier_idwt_pipeline[1])
		return PYROWAVE_SUCCESS;
	const char *canonical = device->precision == 0 ? idwt_fp16_msl_source :
	                        device->precision == 1 ? idwt_fp16_storage_msl_source : idwt_msl_source;
	std::string source;
	try
	{
		if (!build_reduced_barrier_idwt_msl(canonical, device->precision, source))
		{
			device->log("Canonical iDWT shader changed; the benchmark barrier experiment needs a fresh synchronization audit.");
			return PYROWAVE_ERROR_SHADER_COMPILATION;
		}
	}
	catch (const std::bad_alloc &)
	{
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;
	}
	auto *library = compile_library(device, source.c_str(), "experimental reduced-barrier idwt");
	if (!library)
		return PYROWAVE_ERROR_SHADER_COMPILATION;
	id<MTLComputePipelineState> pipelines[2];
	for (int i = 0; i < 2; i++)
	{
		pipelines[i] = create_pipeline_bool_constant(device, library, "pyrowave_idwt_reduced_barriers",
		                                            IdwtThreadgroupSize, 0, i != 0);
		if (!pipelines[i])
			return PYROWAVE_ERROR_SHADER_COMPILATION;
	}
	for (int i = 0; i < 2; i++)
		device->bench_reduced_barrier_idwt_pipeline[i] = pipelines[i];
	return PYROWAVE_SUCCESS;
}
#endif

bool WaveletPyramid::init(pyrowave_device device, const BlockLayout &layout)
{
	const MTLPixelFormat format = wavelet_format(device->precision);

	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType2DArray;
	desc.pixelFormat = format;
	desc.width = layout.aligned_width / 2;
	desc.height = layout.aligned_height / 2;
	desc.arrayLength = NumFrequencyBandsPerLevel * NumComponents;
	desc.mipmapLevelCount = DecompositionLevels;
	// PixelFormatView is required to take the per component/per level views below.
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
	             MTLTextureUsagePixelFormatView;
	desc.storageMode = MTLStorageModePrivate;

	texture = [device->mtl newTextureWithDescriptor:desc];
	if (!texture)
	{
		device->log("Failed to allocate wavelet texture.");
		return false;
	}

	texture.label = @"pyrowave-wavelet";

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			component_layer_views[component][level] =
					[texture newTextureViewWithPixelFormat:format
					                          textureType:MTLTextureType2DArray
					                               levels:NSMakeRange(level, 1)
					                               slices:NSMakeRange(NumFrequencyBandsPerLevel * component,
					                                                  NumFrequencyBandsPerLevel)];

			component_ll_views[component][level] =
					[texture newTextureViewWithPixelFormat:format
					                          textureType:MTLTextureType2D
					                               levels:NSMakeRange(level, 1)
					                               slices:NSMakeRange(NumFrequencyBandsPerLevel * component, 1)];

			if (!component_layer_views[component][level] || !component_ll_views[component][level])
			{
				device->log("Failed to create wavelet texture views.");
				return false;
			}
		}
	}

	return true;
}
}

using namespace PyroWave;

void pyrowave_device_opaque::log(const char *fmt, ...) const
{
	char buffer[512];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);

	if (message_cb)
		message_cb(message_userdata, buffer);
	else
		fprintf(stderr, "pyrowave: %s\n", buffer);
}

namespace
{
bool device_is_supported(id<MTLDevice> mtl)
{
	if (!mtl)
		return false;
	// Apple7 (M1 / A14) and up. This guarantees a 32 wide SIMD group, which the
	// dequant shader's subgroup fast path depends on.
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if (![mtl supportsFamily:MTLGPUFamilyApple7])
            return false;
    } else {
        return false;
    }
	if (mtl.maxThreadsPerThreadgroup.width < AnalyzeFinalizeThreadgroupSize)
		return false;
	return true;
}
}

//////
// Public API

void pyrowave_get_api_version(uint32_t *major, uint32_t *minor, uint32_t *patch)
{
	if (major)
		*major = PYROWAVE_API_VERSION_MAJOR;
	if (minor)
		*minor = PYROWAVE_API_VERSION_MINOR;
	if (patch)
		*patch = PYROWAVE_API_VERSION_PATCH;
}

const char *pyrowave_result_to_string(pyrowave_result result)
{
	return result_string(result);
}

bool pyrowave_device_is_supported(pyrowave_mtl_device mtl_device)
{
	return device_is_supported((__bridge id<MTLDevice>)mtl_device);
}

pyrowave_result pyrowave_create_default_device(pyrowave_device *device)
{
	pyrowave_device_create_info info = {};
	return pyrowave_device_create(&info, device);
}

pyrowave_result pyrowave_device_create(const pyrowave_device_create_info *info, pyrowave_device *device)
{
	if (!info || !device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	@autoreleasepool
	{
		id<MTLDevice> mtl = (__bridge id<MTLDevice>)info->mtl_device;
		if (mtl == nil)
			mtl = MTLCreateSystemDefaultDevice();
		if (!device_is_supported(mtl))
			return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;

		auto created = std::unique_ptr<pyrowave_device_opaque>(new (std::nothrow) pyrowave_device_opaque);
		if (!created)
			return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

		created->message_cb = info->message_callback;
		created->message_userdata = info->message_userdata;
		created->mtl = mtl;
		created->precision = requested_precision();

		id<MTLLibrary> dequant_library =
				compile_library(created.get(), wavelet_dequant_msl_source, "wavelet_dequant");

		char idwt_label[32];
		snprintf(idwt_label, sizeof(idwt_label), "idwt (precision %d)", created->precision);
		const char *idwt_source;
		switch (created->precision)
		{
		case 0: idwt_source = idwt_fp16_msl_source; break;
		case 1: idwt_source = idwt_fp16_storage_msl_source; break;
		default: idwt_source = idwt_msl_source; break;
		}
		id<MTLLibrary> idwt_library = compile_library(created.get(), idwt_source, idwt_label);

		if (!dequant_library || !idwt_library)
			return PYROWAVE_ERROR_SHADER_COMPILATION;

		created->dequant_pipeline = create_pipeline(created.get(), dequant_library,
		                                           "pyrowave_wavelet_dequant", DequantThreadgroupSize);

		for (int i = 0; i < 2 && created->dequant_pipeline; i++)
		{
			created->idwt_pipeline[i] = create_pipeline_bool_constant(created.get(), idwt_library,
			                                                         "pyrowave_idwt", IdwtThreadgroupSize,
			                                                         0, i != 0);
			if (!created->idwt_pipeline[i])
				break;
		}

		if (!created->dequant_pipeline || !created->idwt_pipeline[0] || !created->idwt_pipeline[1])
			return PYROWAVE_ERROR_SHADER_COMPILATION;

		// The dequant shader's subgroup path assumes a 32 wide SIMD group.
		if (created->dequant_pipeline.threadExecutionWidth != 32)
		{
			created->log("Unexpected SIMD width %u, expected 32.",
			             unsigned(created->dequant_pipeline.threadExecutionWidth));
			return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
		}

		MTLSamplerDescriptor *sampler_desc = [MTLSamplerDescriptor new];
		sampler_desc.minFilter = MTLSamplerMinMagFilterNearest;
		sampler_desc.magFilter = MTLSamplerMinMagFilterNearest;
		sampler_desc.mipFilter = MTLSamplerMipFilterNearest;
		sampler_desc.sAddressMode = MTLSamplerAddressModeMirrorRepeat;
		sampler_desc.tAddressMode = MTLSamplerAddressModeMirrorRepeat;
		sampler_desc.rAddressMode = MTLSamplerAddressModeMirrorRepeat;
		created->mirror_repeat_sampler = [created->mtl newSamplerStateWithDescriptor:sampler_desc];

		if (!created->mirror_repeat_sampler)
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

		*device = created.release();
		return PYROWAVE_SUCCESS;
	}
}

void pyrowave_device_destroy(pyrowave_device device)
{
	delete device;
}
