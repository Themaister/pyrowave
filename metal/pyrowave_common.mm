// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Device object, shader helpers and wavelet pyramid shared by the Metal encoder
// and decoder. Objective-C++ under ARC.

#include "pyrowave_common.hpp"

#include "shaders/pyrowave_msl.h"

#include <memory>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Off by default: collection costs a pair of driver clock reads and a couple of
// small allocations per submission.
static bool timestamps_enabled_by_default()
{
	const char *env = getenv("PYROWAVE_TIMESTAMPS");
	return env && *env && strcmp(env, "0") != 0;
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

//////
// GPU timing

TimestampBatch::TimestampBatch(std::shared_ptr<DeviceTimestamps> owner_)
	: owner(std::move(owner_))
{
}

id<MTLCounterSampleBuffer> TimestampBatch::add_pass(const char *tag)
{
	auto buffer = owner->acquire_sample_buffer();
	if (buffer)
		passes.push_back({ buffer, tag });
	return buffer;
}

void TimestampBatch::submit(id<MTLCommandBuffer> cmd)
{
	if (passes.empty())
		return;

	// The block owns the batch, so resolving does not depend on the caller, the
	// encoder or the device still existing when the GPU lands.
	auto self = shared_from_this();
	[cmd addCompletedHandler:^(id<MTLCommandBuffer>) { self->resolve(); }];
}

void TimestampBatch::resolve()
{
	MTLTimestamp cpu_end = 0, gpu_end = 0;
	owner->sample_correlation(&cpu_end, &gpu_end);

	// CPU timestamps are nanoseconds, GPU ticks have no documented unit, so scale
	// by the ratio across this submission. A degenerate window assumes nanoseconds.
	double seconds_per_tick = 1e-9;
	if (gpu_end > gpu_begin && cpu_end > cpu_begin)
		seconds_per_tick = double(cpu_end - cpu_begin) / (double(gpu_end - gpu_begin) * 1e9);

	for (auto &pass : passes)
	{
		// resolveCounterRange: returns autoreleased data, on a Metal thread.
		@autoreleasepool
		{
			NSData *data = [pass.buffer resolveCounterRange:NSMakeRange(0, 2)];
			if (data && data.length >= 2 * sizeof(MTLCounterResultTimestamp))
			{
				auto *samples = static_cast<const MTLCounterResultTimestamp *>(data.bytes);
				// MTLCounterErrorValue marks a sample the GPU never wrote.
				if (samples[0].timestamp != MTLCounterErrorValue &&
				    samples[1].timestamp != MTLCounterErrorValue &&
				    samples[1].timestamp >= samples[0].timestamp)
				{
					owner->accumulate(pass.tag,
					                  double(samples[1].timestamp - samples[0].timestamp) * seconds_per_tick);
				}
			}
		}

		owner->recycle_sample_buffer(pass.buffer);
	}

	passes.clear();
}

DeviceTimestamps::DeviceTimestamps(id<MTLDevice> mtl_)
	: mtl(mtl_), enabled(timestamps_enabled_by_default())
{
	// Without pass boundary sampling there is nothing to attach a buffer to.
	if (![mtl supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
		return;

	for (id<MTLCounterSet> set in mtl.counterSets)
	{
		if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
		{
			counter_set = set;
			break;
		}
	}
}

id<MTLCounterSampleBuffer> DeviceTimestamps::acquire_sample_buffer()
{
	if (!counter_set)
		return nil;

	{
		std::lock_guard<std::mutex> holder{lock};
		if (!pool.empty())
		{
			auto buffer = pool.back();
			pool.pop_back();
			return buffer;
		}
	}

	auto *desc = [MTLCounterSampleBufferDescriptor new];
	desc.counterSet = counter_set;
	desc.storageMode = MTLStorageModeShared;
	desc.sampleCount = 2;

	NSError *error = nil;
	// Failure is not fatal: that one pass goes untimed.
	return [mtl newCounterSampleBufferWithDescriptor:desc error:&error];
}

void DeviceTimestamps::recycle_sample_buffer(id<MTLCounterSampleBuffer> buffer)
{
	std::lock_guard<std::mutex> holder{lock};
	pool.push_back(buffer);
}

void DeviceTimestamps::sample_correlation(MTLTimestamp *cpu, MTLTimestamp *gpu) const
{
	[mtl sampleTimestamps:cpu gpuTimestamp:gpu];
}

std::shared_ptr<TimestampBatch> DeviceTimestamps::begin_batch()
{
	if (!collecting() || !counter_set)
		return nullptr;

	auto batch = std::make_shared<TimestampBatch>(shared_from_this());
	sample_correlation(&batch->cpu_begin, &batch->gpu_begin);
	return batch;
}

void DeviceTimestamps::accumulate(const char *tag, double seconds)
{
	std::lock_guard<std::mutex> holder{lock};
	auto &entry = entries[tag];
	entry.total_time += seconds;
	entry.iterations++;
}

void DeviceTimestamps::report(pyrowave_message_cb cb, void *userdata, bool reset)
{
	// Snapshot, so the callback does not run under the lock.
	std::map<std::string, Entry> snapshot;
	{
		std::lock_guard<std::mutex> holder{lock};
		snapshot = entries;
		if (reset)
			entries.clear();
	}

	// Arm collection, so polling this alone works without PYROWAVE_TIMESTAMPS.
	const bool was_collecting = enabled.exchange(true, std::memory_order_relaxed);

	for (auto &entry : snapshot)
	{
		char msg[256];
		snprintf(msg, sizeof(msg), "%s: %.3f ms per iteration (%llu iterations)",
		         entry.first.c_str(),
		         1e3 * entry.second.total_time / double(entry.second.iterations),
		         static_cast<unsigned long long>(entry.second.iterations));
		cb(userdata, msg);
	}

	if (snapshot.empty())
	{
		cb(userdata, was_collecting ?
		             "No GPU timings have been collected yet." :
		             "GPU timing was off. It is now on; figures appear after the next "
		             "encode or decode completes.");
	}
}

id<MTLComputeCommandEncoder> begin_compute_pass(id<MTLCommandBuffer> cmd, MTLDispatchType dispatch_type,
                                                TimestampBatch *batch, const char *tag)
{
	if (batch)
	{
		if (auto sample_buffer = batch->add_pass(tag))
		{
			// Owned, not autoreleased: no encode or decode path has a pool.
			auto *desc = [MTLComputePassDescriptor new];
			desc.dispatchType = dispatch_type;

			auto *attachment = desc.sampleBufferAttachments[0];
			attachment.sampleBuffer = sample_buffer;
			attachment.startOfEncoderSampleIndex = 0;
			attachment.endOfEncoderSampleIndex = 1;

			return [cmd computeCommandEncoderWithDescriptor:desc];
		}
	}

	return [cmd computeCommandEncoderWithDispatchType:dispatch_type];
}

id<MTLBlitCommandEncoder> begin_blit_pass(id<MTLCommandBuffer> cmd, TimestampBatch *batch, const char *tag)
{
	if (batch)
	{
		if (auto sample_buffer = batch->add_pass(tag))
		{
			auto *desc = [MTLBlitPassDescriptor new];

			auto *attachment = desc.sampleBufferAttachments[0];
			attachment.sampleBuffer = sample_buffer;
			attachment.startOfEncoderSampleIndex = 0;
			attachment.endOfEncoderSampleIndex = 1;

			return [cmd blitCommandEncoderWithDescriptor:desc];
		}
	}

	return [cmd blitCommandEncoder];
}

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
// Lets a NULL callback fall back to the device's message sink.
void log_to_device(void *userdata, const char *msg)
{
	static_cast<pyrowave_device>(userdata)->log("%s", msg);
}

bool device_is_supported(id<MTLDevice> mtl)
{
	if (!mtl)
		return false;
	// Apple7 (M1 / A14) and up. This guarantees a 32 wide SIMD group, which the
	// dequant shader's subgroup fast path depends on.
	if (![mtl supportsFamily:MTLGPUFamilyApple7])
		return false;
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
		created->timestamps = std::make_shared<DeviceTimestamps>(mtl);

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

void pyrowave_device_report_performance_stats(pyrowave_device device, pyrowave_message_cb cb, void *userdata, bool reset)
{
	if (!device)
		return;

	if (!cb)
	{
		cb = log_to_device;
		userdata = device;
	}

	device->timestamps->report(cb, userdata, reset);

	if (!device->timestamps->counters_supported())
		cb(userdata, "GPU pass timestamps are not supported by this device.");

	@autoreleasepool
	{
		// Metal has no per heap budget like VK_EXT_memory_budget, only this
		// process's allocation total against the recommended working set.
		if (@available(macOS 10.15, iOS 16.0, tvOS 16.0, *))
		{
			char msg[256];
			snprintf(msg, sizeof(msg),
			         "Memory (%s): CurrentAllocated %.3f MiB, RecommendedMaxWorkingSet %.3f MiB",
			         device->mtl.hasUnifiedMemory ? "unified" : "discrete",
			         double(device->mtl.currentAllocatedSize) / (1024.0 * 1024.0),
			         double(device->mtl.recommendedMaxWorkingSetSize) / (1024.0 * 1024.0));
			cb(userdata, msg);
		}
	}
}

void pyrowave_device_destroy(pyrowave_device device)
{
	delete device;
}
