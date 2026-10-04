// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>

#include "pyrowave_metal.h"
#include "pyrowave_bench.h"
#include "experimental_render.msl.h"
#include <mach/mach_time.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

struct Options
{
	int width = 1920;
	int height = 1080;
	bool chroma_420 = true;
	size_t frames = 1000;
	size_t warmup = 30;
	size_t fps = 0;
	size_t bytes = 500000;
	int precision = -1;
	bool gpu_input = true;
	bool decode_only = false;
	bool private_output = false;
	bool profile = false;
	bool batched_dequant = false;
	bool reduced_idwt_barriers = false;
	bool native_dequant = false;
	bool hybrid_dequant = false;
	bool native_idwt = false;
	bool fused_idwt = false;
	std::string render = "none";
	int worker_qos = -1;
	std::string compare;
	std::string samples_path;
	std::string input;
};

void usage(const char *program)
{
	std::printf("Usage: %s [options]\n"
	            "  --width N             Synthetic frame width (default 1920)\n"
	            "  --height N            Synthetic frame height (default 1080)\n"
	            "  --chroma 420|444      Synthetic chroma subsampling (default 420)\n"
	            "  --frames N            Measured frames per variant (default 1000)\n"
	            "  --warmup N            Warmup frames per variant (default 30)\n"
	            "  --fps N               Pace at N frames/s (default: run unpaced)\n"
	            "  --decode-only         Encode/packetize once in setup, then only decode\n"
	            "  --output-storage shared|private (default shared)\n"
	            "  --batched-dequant     Experimental band batching\n"
	            "  --reduced-idwt-barriers  Experimental redundant apron barrier removal\n"
	            "  --native-dequant      Experimental packed bit-plane unpacking\n"
	            "  --hybrid-dequant      Experimental short-plane unpacking\n"
	            "  --native-idwt         Experimental balanced reconstruction loads\n"
	            "  --fused-idwt          Diagnostic two-level fusion (not bit-identical)\n"
	            "  --render none|wait|chain  Offscreen BT.709 RGB render (default none)\n"
	            "  --profile             Include CPU phases and GPU pass counters\n"
	            "  --compare output|dequant|idwt|native-dequant|hybrid-dequant|native-idwt|fused-idwt|compact-idwt|combined|optimized|native-batched|render|render-fused|render-optimized\n"
	            "                        Interleave variants on one cached bitstream\n"
	            "  --qos default|interactive (default inherited)\n"
	            "  --samples FILE.csv    Save measured samples after timing\n"
	            "  --bytes N             Encoder frame budget in bytes (default 500000)\n"
	            "  --precision 0|1|2     FP16 / FP32 math+FP16 storage / FP32\n"
	            "                        Default: PYROWAVE_PRECISION, otherwise 1\n"
	            "  --input FILE.y4m      Repeat the first 8-bit 420/444 frame in memory;\n"
	            "                        its header supplies dimensions and chroma\n"
	            "  --input-mode gpu|cpu  Resident IOSurface / CPU upload (default gpu)\n"
	            "  --help                Show this help\n", program);
}

size_t number(const std::string &value, const char *name, size_t maximum)
{
	if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
		throw std::runtime_error(std::string(name) + " requires an unsigned decimal integer");
	errno = 0;
	char *end = nullptr;
	const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
	if (errno == ERANGE || !end || *end || parsed > maximum)
		throw std::runtime_error(std::string(name) + " is out of range");
	return size_t(parsed);
}

bool parse_options(int argc, char **argv, Options &options)
{
	for (int i = 1; i < argc; i++)
	{
		const std::string arg = argv[i];
		if (arg == "--help")
		{
			usage(argv[0]);
			return false;
		}
		if (arg == "--decode-only")
		{
			options.decode_only = true;
			continue;
		}
		if (arg == "--profile" || arg == "--batched-dequant" || arg == "--reduced-idwt-barriers")
		{
			if (arg == "--profile") options.profile = true;
			else if (arg == "--batched-dequant") options.batched_dequant = true;
			else options.reduced_idwt_barriers = true;
			continue;
		}
		if (arg == "--native-dequant" || arg == "--hybrid-dequant" || arg == "--native-idwt" || arg == "--fused-idwt")
		{
			if (arg == "--native-dequant") options.native_dequant = true;
			else if (arg == "--hybrid-dequant") options.hybrid_dequant = true;
			else if (arg == "--native-idwt") options.native_idwt = true;
			else options.fused_idwt = true;
			continue;
		}
		if (i + 1 == argc)
			throw std::runtime_error("Missing value for " + arg);
		const std::string value = argv[++i];
		if (arg == "--width")
			options.width = int(number(value, "--width", 16384));
		else if (arg == "--height")
			options.height = int(number(value, "--height", 16384));
		else if (arg == "--chroma")
		{
			if (value != "420" && value != "444")
				throw std::runtime_error("--chroma must be 420 or 444");
			options.chroma_420 = value == "420";
		}
		else if (arg == "--frames")
			options.frames = number(value, "--frames", std::numeric_limits<int>::max());
		else if (arg == "--warmup")
			options.warmup = number(value, "--warmup", std::numeric_limits<int>::max());
		else if (arg == "--fps")
		{
			options.fps = number(value, "--fps", 1000);
			if (!options.fps)
				throw std::runtime_error("--fps must be in [1, 1000]");
		}
		else if (arg == "--bytes")
			options.bytes = number(value, "--bytes", UINT32_MAX);
		else if (arg == "--precision")
			options.precision = int(number(value, "--precision", 2));
		else if (arg == "--input")
			options.input = value;
		else if (arg == "--input-mode")
		{
			if (value != "gpu" && value != "cpu")
				throw std::runtime_error("--input-mode must be gpu or cpu");
			options.gpu_input = value == "gpu";
		}
		else if (arg == "--output-storage")
		{
			if (value != "shared" && value != "private")
				throw std::runtime_error("--output-storage must be shared or private");
			options.private_output = value == "private";
		}
		else if (arg == "--compare")
		{
			if (value != "output" && value != "dequant" && value != "idwt" &&
			    value != "native-dequant" && value != "hybrid-dequant" && value != "native-idwt" && value != "fused-idwt" &&
			    value != "compact-idwt" && value != "combined" && value != "optimized" && value != "native-batched" &&
			    value != "render" && value != "render-fused" && value != "render-optimized")
				throw std::runtime_error("Unknown --compare variant");
			options.compare = value;
		}
		else if (arg == "--render")
		{
			if (value != "none" && value != "wait" && value != "chain")
				throw std::runtime_error("--render must be none, wait, or chain");
			options.render = value;
		}
		else if (arg == "--samples") options.samples_path = value;
		else if (arg == "--qos")
		{
			if (value != "default" && value != "interactive")
				throw std::runtime_error("--qos must be default or interactive");
			options.worker_qos = value == "interactive" ? 1 : 0;
		}
		else
			throw std::runtime_error("Unknown option: " + arg);
	}
	if (!options.frames)
		throw std::runtime_error("--frames must be at least 1");
	if (options.bytes < 4)
		throw std::runtime_error("--bytes must be at least 4");
	if (!options.compare.empty() && (!options.decode_only || options.profile || options.batched_dequant || options.reduced_idwt_barriers ||
	                              options.native_dequant || options.native_idwt || options.fused_idwt))
		throw std::runtime_error("--compare requires --decode-only and cannot combine with profiling or shader override flags");
	if (!options.compare.empty() && options.hybrid_dequant)
		throw std::runtime_error("--compare cannot combine with --hybrid-dequant");
	if (options.native_dequant && options.hybrid_dequant)
		throw std::runtime_error("Select only one unpacking override");
	if (options.compare == "render" || options.compare == "render-fused" || options.compare == "render-optimized")
		options.render = "chain";
	if (options.precision < 0)
	{
		const char *env = std::getenv("PYROWAVE_PRECISION");
		options.precision = env ? int(number(env, "PYROWAVE_PRECISION", 2)) : 1;
	}
	return true;
}

void validate_dimensions(const Options &options)
{
	if (options.width < 1 || options.height < 1 || options.width > 16384 || options.height > 16384)
		throw std::runtime_error("Frame dimensions must be in [1, 16384]");
	if (options.chroma_420 && ((options.width & 1) || (options.height & 1)))
		throw std::runtime_error("420 chroma requires even width and height");
}

struct Plane
{
	int width = 0;
	int height = 0;
	std::vector<uint8_t> pixels;
};

struct Frame
{
	Plane planes[3];

	void allocate(const Options &options)
	{
		for (int i = 0; i < 3; i++)
		{
			planes[i].width = i && options.chroma_420 ? options.width / 2 : options.width;
			planes[i].height = i && options.chroma_420 ? options.height / 2 : options.height;
			planes[i].pixels.resize(size_t(planes[i].width) * size_t(planes[i].height));
		}
	}
};

std::string read_line(FILE *file)
{
	std::string line;
	for (;;)
	{
		const int c = std::fgetc(file);
		if (c == EOF)
			throw std::runtime_error("Unexpected end of Y4M file while reading a header");
		if (c == '\n')
			break;
		if (line.size() == 16384)
			throw std::runtime_error("Y4M header exceeds 16384 bytes");
		line += char(c);
	}
	if (!line.empty() && line.back() == '\r')
		line.pop_back();
	return line;
}

void read_first_frame(Options &options, Frame &frame)
{
	using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
	File file(std::fopen(options.input.c_str(), "rb"), std::fclose);
	if (!file)
		throw std::runtime_error("Cannot open input: " + options.input);
	std::istringstream header(read_line(file.get()));
	std::string token;
	header >> token;
	if (token != "YUV4MPEG2")
		throw std::runtime_error("Input is not a YUV4MPEG2 file");
	bool have_width = false, have_height = false, have_chroma = false;
	// The Y4M default when C is omitted is 8-bit 420jpeg.
	options.chroma_420 = true;
	while (header >> token)
	{
		if (token[0] == 'W')
		{
			if (have_width)
				throw std::runtime_error("Duplicate width in Y4M header");
			options.width = int(number(token.substr(1), "Y4M width", 16384));
			have_width = true;
		}
		else if (token[0] == 'H')
		{
			if (have_height)
				throw std::runtime_error("Duplicate height in Y4M header");
			options.height = int(number(token.substr(1), "Y4M height", 16384));
			have_height = true;
		}
		else if (token[0] == 'C')
		{
			if (have_chroma)
				throw std::runtime_error("Duplicate chroma format in Y4M header");
			if (token == "C444")
				options.chroma_420 = false;
			else if (token != "C420" && token != "C420jpeg" && token != "C420mpeg2" && token != "C420paldv")
				throw std::runtime_error("Unsupported Y4M chroma: " + token + "; only 8-bit 420 and 444 are supported");
			have_chroma = true;
		}
	}
	if (!have_width || !have_height)
		throw std::runtime_error("Y4M header must specify width and height");
	validate_dimensions(options);
	std::istringstream frame_header(read_line(file.get()));
	frame_header >> token;
	if (token != "FRAME")
		throw std::runtime_error("Missing first FRAME header in Y4M input");
	frame.allocate(options);
	for (auto &plane : frame.planes)
		if (std::fread(plane.pixels.data(), 1, plane.pixels.size(), file.get()) != plane.pixels.size())
			throw std::runtime_error("Truncated first frame in Y4M input");
}

void make_synthetic(Frame &frame)
{
	for (int i = 0; i < 3; i++)
	{
		auto &plane = frame.planes[i];
		for (int y = 0; y < plane.height; y++)
		{
			for (int x = 0; x < plane.width; x++)
			{
				uint32_t hash = uint32_t(x) * 1664525u + uint32_t(y) * 1013904223u + uint32_t(i + 1) * 0x9e3779b9u;
				hash = (hash ^ (hash >> 13)) * 1274126177u;
				int value;
				if (y < plane.height / 2 && x < plane.width / 2)
					value = (x * 193 / plane.width + y * 131 / plane.height + 23 * i) & 255;
				else if (y < plane.height / 2)
					value = ((x / 8 + y / 8) & 1) ? 224 : 32;
				else if (x < plane.width / 2)
					value = (x % 32 < 3 || y % 24 < 3 || ((x / 12) & 1 && y % 24 < 15)) ? 48 : 208;
				else
					value = int((hash ^ (hash >> 16)) & 255);
				plane.pixels[size_t(y) * size_t(plane.width) + size_t(x)] = uint8_t(i ? 64 + value / 2 : value);
			}
		}
	}
}

void check(pyrowave_result result, const char *operation)
{
	if (result != PYROWAVE_SUCCESS)
		throw std::runtime_error(std::string(operation) + ": " + pyrowave_result_to_string(result));
}

struct Resources
{
	pyrowave_device device = nullptr;
	pyrowave_encoder encoder = nullptr;
	pyrowave_decoder decoder = nullptr;
	IOSurfaceRef surfaces[3] = {};
	id<MTLDevice> metal_device;
	id<MTLCommandQueue> queue;
	id<MTLTexture> output[3];
	id<MTLTexture> alternate_output[3];
	id<MTLRenderPipelineState> render_pipeline;
	id<MTLTexture> render_output[2];
	size_t variant = 0;
	bool render_wait = false;
	bool render_fused = false;
	pyrowave_cpu_buffer cpu_input = {};
	pyrowave_gpu_input gpu_input = {};
	pyrowave_gpu_buffers gpu_output = {};
	std::vector<uint8_t> packet_data;
	std::vector<pyrowave_packet> packets;
	size_t packet_count = 0;

	~Resources()
	{
		pyrowave_decoder_destroy(decoder);
		pyrowave_encoder_destroy(encoder);
		pyrowave_device_destroy(device);
		for (auto surface : surfaces)
			if (surface)
				CFRelease(surface);
	}
};

void create_surface(Resources &resources, const Plane &plane, int index)
{
	const size_t stride = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, size_t(plane.width));
	NSDictionary *properties = @{
		(__bridge NSString *)kIOSurfaceWidth: @(plane.width),
		(__bridge NSString *)kIOSurfaceHeight: @(plane.height),
		(__bridge NSString *)kIOSurfaceBytesPerElement: @1,
		(__bridge NSString *)kIOSurfaceBytesPerRow: @(stride),
		(__bridge NSString *)kIOSurfaceAllocSize: @(stride * size_t(plane.height)),
		(__bridge NSString *)kIOSurfacePixelFormat: @((uint32_t('L') << 24) | (uint32_t('0') << 16) | (uint32_t('0') << 8) | uint32_t('8'))
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
	if (!surface)
		throw std::runtime_error("Failed to create input IOSurface");
	resources.surfaces[index] = surface;
	resources.gpu_input.planes[index] = surface;
	if (IOSurfaceLock(surface, 0, nullptr) != 0)
		throw std::runtime_error("Failed to lock input IOSurface");
	auto *pixels = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface));
	const size_t actual_stride = IOSurfaceGetBytesPerRow(surface);
	if (!pixels || actual_stride < size_t(plane.width))
	{
		IOSurfaceUnlock(surface, 0, nullptr);
		throw std::runtime_error("Invalid input IOSurface allocation");
	}
	for (int y = 0; y < plane.height; y++)
		std::memcpy(pixels + size_t(y) * actual_stride,
		            plane.pixels.data() + size_t(y) * size_t(plane.width), size_t(plane.width));
	if (IOSurfaceUnlock(surface, 0, nullptr) != 0)
		throw std::runtime_error("Failed to unlock input IOSurface");
}

void encode(Resources &resources, const Options &options)
{
	const pyrowave_rate_control rate = { options.bytes };
	if (options.gpu_input)
		check(pyrowave_encoder_encode_gpu_synchronous(resources.encoder, &resources.gpu_input, &rate), "GPU-input encode");
	else
		check(pyrowave_encoder_encode_cpu_synchronous(resources.encoder, &resources.cpu_input, &rate), "CPU-input encode");
}

void finish_encode(Resources &resources, size_t &raw_size, size_t &metadata_size)
{
	const void *raw = nullptr, *metadata = nullptr;
	// The encode entry point submits without waiting. This query is its completion fence.
	check(pyrowave_encoder_get_mapped_raw_bitstream(resources.encoder, &raw, &raw_size, &metadata, &metadata_size),
	      "Wait for encoded frame");
}

void setup_render(Resources &resources, const Options &options)
{
	if (options.render == "none") return;
	resources.render_wait = options.render == "wait";
	NSError *error = nil;
	auto source = PyroWave::bench_render_source();
	auto library = [resources.metal_device newLibraryWithSource:@(source.c_str()) options:nil error:&error];
	if (!library) throw std::runtime_error(std::string("Render shader compilation: ") + error.localizedDescription.UTF8String);
	auto descriptor = [MTLRenderPipelineDescriptor new];
	descriptor.vertexFunction = [library newFunctionWithName:@"bench_fullscreen"];
	descriptor.fragmentFunction = [library newFunctionWithName:@"bench_rgb_fragment"];
	descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	resources.render_pipeline = [resources.metal_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
	if (!resources.render_pipeline) throw std::runtime_error(std::string("Render pipeline: ") + error.localizedDescription.UTF8String);
	auto texture = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                width:options.width height:options.height mipmapped:NO];
	texture.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite;
	texture.storageMode = MTLStorageModeShared;
	for (int i = 0; i < (options.compare.empty() ? 1 : 2); i++)
	{
		resources.render_output[i] = [resources.metal_device newTextureWithDescriptor:texture];
		if (!resources.render_output[i]) throw std::runtime_error("Failed to allocate RGB output");
	}
}

void encode_render(Resources &resources, id<MTLCommandBuffer> command)
{
	auto descriptor = [MTLRenderPassDescriptor renderPassDescriptor];
	descriptor.colorAttachments[0].texture = resources.render_output[resources.variant];
	descriptor.colorAttachments[0].loadAction = MTLLoadActionDontCare;
	descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
	auto encoder = [command renderCommandEncoderWithDescriptor:descriptor];
	if (!encoder) throw std::runtime_error("Failed to create RGB render encoder");
	[encoder setRenderPipelineState:resources.render_pipeline];
	for (int i = 0; i < 3; i++)
		[encoder setFragmentTexture:(__bridge id<MTLTexture>)resources.gpu_output.planes[i] atIndex:i];
	[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[encoder endEncoding];
}

void setup(Resources &resources, const Options &options, Frame &frame)
{
	const std::string precision = std::to_string(options.precision);
	if (setenv("PYROWAVE_PRECISION", precision.c_str(), 1) != 0)
		throw std::runtime_error("Failed to set PYROWAVE_PRECISION");
	resources.metal_device = MTLCreateSystemDefaultDevice();
	if (!resources.metal_device)
		throw std::runtime_error("No Metal device is available; run from a normal macOS Terminal with GPU access (a sandbox may hide the GPU)");
	pyrowave_device_create_info device_info = {};
	device_info.mtl_device = (__bridge void *)resources.metal_device;
	check(pyrowave_device_create(&device_info, &resources.device), "Create Metal device");
	const auto chroma = options.chroma_420 ? PYROWAVE_CHROMA_SUBSAMPLING_420 : PYROWAVE_CHROMA_SUBSAMPLING_444;
	const pyrowave_encoder_create_info encoder_info = { resources.device, options.width, options.height, chroma };
	check(pyrowave_encoder_create(&encoder_info, &resources.encoder), "Create encoder");
	const pyrowave_decoder_create_info decoder_info = { resources.device, options.width, options.height, chroma };
	check(pyrowave_decoder_create(&decoder_info, &resources.decoder), "Create decoder");
	resources.queue = [resources.metal_device newCommandQueue];
	if (!resources.queue)
		throw std::runtime_error("Failed to create decode command queue");
	resources.cpu_input.width = options.width;
	resources.cpu_input.height = options.height;
	resources.cpu_input.format = options.chroma_420 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV420P : PYROWAVE_CPU_BUFFER_FORMAT_YUV444P;
	for (int i = 0; i < 3; i++)
	{
		auto &plane = frame.planes[i];
		resources.cpu_input.data[i] = plane.pixels.data();
		resources.cpu_input.row_stride_in_bytes[i] = size_t(plane.width);
		resources.cpu_input.plane_size_in_bytes[i] = plane.pixels.size();
		if (options.gpu_input)
			create_surface(resources, plane, i);
		auto *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
		                                                                         width:plane.width height:plane.height mipmapped:NO];
		descriptor.usage = MTLTextureUsageShaderWrite | (options.render != "none" ? MTLTextureUsageShaderRead : 0);
		descriptor.storageMode = options.private_output ? MTLStorageModePrivate : MTLStorageModeShared;
		resources.output[i] = [resources.metal_device newTextureWithDescriptor:descriptor];
		if (!resources.output[i])
			throw std::runtime_error("Failed to create decode output texture");
		resources.gpu_output.planes[i] = (__bridge void *)resources.output[i];
		if (!options.compare.empty())
		{
			if (options.compare == "output")
				descriptor.storageMode = options.private_output ? MTLStorageModeShared : MTLStorageModePrivate;
			resources.alternate_output[i] = [resources.metal_device newTextureWithDescriptor:descriptor];
			if (!resources.alternate_output[i])
				throw std::runtime_error("Failed to create alternate output texture");
		}
	}
	// Prime input and result allocations even with --warmup 0. Subsequent iterations reuse them.
	encode(resources, options);
	size_t raw_size = 0, metadata_size = 0, count = 0;
	finish_encode(resources, raw_size, metadata_size);
	if (metadata_size > SIZE_MAX - 8 || raw_size > SIZE_MAX - metadata_size - 8)
		throw std::runtime_error("Encoded buffer sizes overflow host address space");
	resources.packet_data.resize(raw_size + metadata_size + 8);
	check(pyrowave_encoder_compute_num_packets(resources.encoder, UINT32_MAX, &count), "Count frame packets");
	if (!count)
		throw std::runtime_error("Encoder produced no frame packets");
	resources.packets.resize(count);
	setup_render(resources, options);
}

double milliseconds(Clock::time_point start, Clock::time_point end)
{
	return std::chrono::duration<double, std::milli>(end - start).count();
}

double host_seconds()
{
	static const double scale = [] {
		mach_timebase_info_data_t info;
		mach_timebase_info(&info);
		return double(info.numer) / double(info.denom) * 1e-9;
	}();
	return double(mach_absolute_time()) * scale;
}

struct Sample
{
	double encode_gpu = 0.0;
	double encode_wall = 0.0;
	double packetize = 0.0;
	double decode_gpu = 0.0;
	double decode_wall = 0.0;
	double roundtrip = 0.0;
	double parse_cpu = 0.0;
	double encode_commands_cpu = 0.0;
	double commit_cpu = 0.0;
	double commit_to_gpu = 0.0;
	double gpu_to_return = 0.0;
	double packet_to_gpu = 0.0;
	pyrowave_bench_decode_timings stages = {
		std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
		std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
		std::numeric_limits<double>::quiet_NaN()
	};
	Clock::time_point completed;
	size_t payload_bytes = 0;
};

void packetize_frame(Resources &resources)
{
	size_t expected = 0, actual = 0;
	check(pyrowave_encoder_compute_num_packets(resources.encoder, UINT32_MAX, &expected), "Count frame packets");
	if (!expected || expected > resources.packets.size())
		throw std::runtime_error("Frame packet count exceeded the reusable allocation");
	check(pyrowave_encoder_packetize(resources.encoder, resources.packets.data(), UINT32_MAX, &actual,
	                                resources.packet_data.data(), resources.packet_data.size()), "Packetize frame");
	if (actual != expected)
		throw std::runtime_error("Packetization did not produce the complete frame");
	resources.packet_count = actual;
}

Sample run_sample(Resources &resources, const Options &options)
{
	Sample sample;
	const auto begin = Clock::now();
	if (!options.decode_only)
	{
		encode(resources, options);
		size_t raw_size = 0, metadata_size = 0;
		finish_encode(resources, raw_size, metadata_size);
		const auto encoded = Clock::now();
		sample.encode_wall = milliseconds(begin, encoded);
		sample.encode_gpu = pyrowave_bench_last_gpu_ms(resources.encoder);

		const auto packet_begin = Clock::now();
		packetize_frame(resources);
		sample.packetize = milliseconds(packet_begin, Clock::now());
	}

	const double decode_begin_host = host_seconds();
	const auto decode_begin = Clock::now();
	pyrowave_decoder_clear(resources.decoder);
	for (size_t i = 0; i < resources.packet_count; i++)
	{
		const auto &packet = resources.packets[i];
		if (!packet.size || packet.offset > resources.packet_data.size() ||
		    packet.size > resources.packet_data.size() - packet.offset)
			throw std::runtime_error("Encoder returned an invalid packet range");
		check(pyrowave_decoder_push_packet(resources.decoder, resources.packet_data.data() + packet.offset, packet.size),
		      "Push frame packet");
		sample.payload_bytes += packet.size;
	}
	if (!pyrowave_decoder_decode_is_ready(resources.decoder, false))
		throw std::runtime_error("Packetized frame is not complete for decoding");
	const auto parsed = Clock::now();
	id<MTLCommandBuffer> command = [resources.queue commandBuffer];
	if (!command)
		throw std::runtime_error("Failed to create decode command buffer");
	check(pyrowave_decoder_decode_gpu_buffer(resources.decoder, (__bridge void *)command, &resources.gpu_output),
	      "Encode decode commands");
	if (options.render != "none" && !resources.render_wait && !resources.render_fused)
		encode_render(resources, command);
	const auto encoded_commands = Clock::now();
	const double commit_host = host_seconds();
	[command commit];
	const auto committed = Clock::now();
	[command waitUntilCompleted];
	double gpu_time = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
	double final_gpu_end = command.GPUEndTime;
	double render_commands_cpu = 0.0, render_commit_cpu = 0.0;
	if (command.status != MTLCommandBufferStatusCompleted)
		throw std::runtime_error(std::string("Decode command buffer failed: ") +
		                         (command.error.localizedDescription.UTF8String ?: "unknown error"));
	if (options.render != "none" && resources.render_wait)
	{
		const auto render_begin = Clock::now();
		auto render_command = [resources.queue commandBuffer];
		if (!render_command) throw std::runtime_error("Failed to create render command buffer");
		encode_render(resources, render_command);
		const auto render_encoded = Clock::now();
		[render_command commit];
		const auto render_committed = Clock::now();
		[render_command waitUntilCompleted];
		if (render_command.status != MTLCommandBufferStatusCompleted)
			throw std::runtime_error(std::string("RGB render command buffer failed: ") +
			                         (render_command.error.localizedDescription.UTF8String ?: "unknown error"));
		render_commands_cpu = milliseconds(render_begin, render_encoded);
		render_commit_cpu = milliseconds(render_encoded, render_committed);
		gpu_time += (render_command.GPUEndTime - render_command.GPUStartTime) * 1000.0;
		final_gpu_end = render_command.GPUEndTime;
	}
	const double return_host = host_seconds();
	const auto decoded = Clock::now();
	sample.completed = decoded;
	sample.decode_gpu = gpu_time;
	sample.parse_cpu = milliseconds(decode_begin, parsed);
	sample.encode_commands_cpu = milliseconds(parsed, encoded_commands) + render_commands_cpu;
	sample.commit_cpu = milliseconds(encoded_commands, committed) + render_commit_cpu;
	sample.commit_to_gpu = command.GPUStartTime > 0.0 && command.GPUStartTime >= commit_host ?
	                       (command.GPUStartTime - commit_host) * 1000.0 : std::numeric_limits<double>::quiet_NaN();
	sample.gpu_to_return = final_gpu_end > 0.0 && return_host >= final_gpu_end ?
	                       (return_host - final_gpu_end) * 1000.0 : std::numeric_limits<double>::quiet_NaN();
	sample.packet_to_gpu = final_gpu_end > decode_begin_host ?
	                       (final_gpu_end - decode_begin_host) * 1000.0 : std::numeric_limits<double>::quiet_NaN();
	sample.decode_wall = milliseconds(decode_begin, decoded);
	sample.roundtrip = options.decode_only ? sample.decode_wall : milliseconds(begin, decoded);
	if (options.profile)
		check(pyrowave_bench_get_decode_timings(resources.decoder, &sample.stages), "Resolve GPU pass timings");
	return sample;
}

struct Metric
{
	std::vector<double> values;

	void add(double value, bool allow_zero = false)
	{
		if (std::isfinite(value) && (value > 0.0 || (allow_zero && value == 0.0)))
			values.push_back(value);
	}

	void report(const char *name, size_t count) const
	{
		if (values.size() != count)
		{
			std::printf("%-24s unavailable (%zu of %zu valid timing samples)\n", name, values.size(), count);
			return;
		}
		auto sorted = values;
		std::sort(sorted.begin(), sorted.end());
		double sum = 0.0;
		for (double value : values)
			sum += value;
		const double average = sum / double(count);
		auto percentile = [&](double p) {
			const double index = p * double(sorted.size() - 1);
			const size_t lower = size_t(std::floor(index)), upper = size_t(std::ceil(index));
			return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - double(lower));
		};
		std::printf("%-24s %10.3f %10.3f %10.3f %10.3f %11.1f\n", name,
		            average, percentile(0.50), percentile(0.95), percentile(0.99), 1000.0 / average);
	}
};

using DecodedPlanes = std::array<std::vector<uint8_t>, 3>;

DecodedPlanes read_output(const Resources &resources, const Frame &frame, id<MTLTexture> const textures[3])
{
	DecodedPlanes decoded;
	for (int i = 0; i < 3; i++)
	{
		const auto &plane = frame.planes[i];
		decoded[i].resize(plane.pixels.size());
		if (textures[i].storageMode == MTLStorageModePrivate)
		{
			const size_t stride = (size_t(plane.width) + 255) & ~size_t(255);
			id<MTLBuffer> readback = [resources.metal_device newBufferWithLength:stride * plane.height
			                                                         options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> command = [resources.queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
			if (!readback || !command || !blit)
				throw std::runtime_error("Failed to prepare private texture readback");
			[blit copyFromTexture:textures[i] sourceSlice:0 sourceLevel:0
			         sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(plane.width, plane.height, 1)
			             toBuffer:readback destinationOffset:0 destinationBytesPerRow:stride
			 destinationBytesPerImage:stride * plane.height];
			[blit endEncoding];
			[command commit];
			[command waitUntilCompleted];
			if (command.status != MTLCommandBufferStatusCompleted)
				throw std::runtime_error("Private texture readback failed");
			for (int y = 0; y < plane.height; y++)
				std::memcpy(decoded[i].data() + size_t(y) * plane.width,
				            static_cast<const uint8_t *>(readback.contents) + size_t(y) * stride, plane.width);
		}
		else
			[textures[i] getBytes:decoded[i].data() bytesPerRow:plane.width
			              fromRegion:MTLRegionMake2D(0, 0, plane.width, plane.height) mipmapLevel:0];
	}
	return decoded;
}

void validate_output(const DecodedPlanes &decoded, const Frame &frame)
{
	std::printf("\nLast-frame output (readback excluded from timing):\n");
	const char *names[] = { "Y", "Cb", "Cr" };
	for (int i = 0; i < 3; i++)
	{
		const auto &plane = frame.planes[i];
		double squared_error = 0.0;
		uint64_t checksum = 14695981039346656037ull;
		for (size_t j = 0; j < decoded[i].size(); j++)
		{
			const int difference = int(decoded[i][j]) - int(plane.pixels[j]);
			squared_error += double(difference * difference);
			checksum = (checksum ^ decoded[i][j]) * 1099511628211ull;
		}
		const double mse = squared_error / double(decoded[i].size());
		const double psnr = mse ? 10.0 * std::log10(255.0 * 255.0 / mse) : std::numeric_limits<double>::infinity();
		std::printf("  %-2s PSNR %8.3f dB  FNV-1a64 %016llx\n", names[i], psnr, static_cast<unsigned long long>(checksum));
	}
}

void select_variant(Resources &resources, const Options &options, size_t variant)
{
	resources.variant = variant;
	for (int i = 0; i < 3; i++)
		resources.gpu_output.planes[i] = (__bridge void *)(variant ? resources.alternate_output[i] : resources.output[i]);
	if (options.compare == "dequant")
		check(pyrowave_bench_set_batched_dequant(resources.decoder, variant != 0), "Select dequant variant");
	else if (options.compare == "idwt")
		check(pyrowave_bench_set_reduced_idwt_barriers(resources.decoder, variant != 0), "Select IDWT variant");
	else if (options.compare == "native-dequant")
		check(pyrowave_bench_set_native_dequant(resources.decoder, variant != 0), "Select native dequant");
	else if (options.compare == "hybrid-dequant")
		check(pyrowave_bench_set_hybrid_dequant(resources.decoder, variant != 0), "Select hybrid dequant");
	else if (options.compare == "native-idwt")
		check(pyrowave_bench_set_native_idwt(resources.decoder, variant != 0), "Select native IDWT");
	else if (options.compare == "fused-idwt")
		check(pyrowave_bench_set_fused_idwt(resources.decoder, variant != 0), "Select fused IDWT");
	else if (options.compare == "compact-idwt")
		check(pyrowave_bench_set_compact_fused_idwt(resources.decoder, variant != 0), "Select compact fused IDWT");
	else if (options.compare == "combined")
	{
		check(pyrowave_bench_set_native_dequant(resources.decoder, variant != 0), "Select packed unpacking");
		check(pyrowave_bench_set_native_idwt(resources.decoder, variant != 0), "Select balanced reconstruction");
		check(pyrowave_bench_set_batched_dequant(resources.decoder, variant != 0), "Select band batching");
	}
	else if (options.compare == "optimized")
	{
		check(pyrowave_bench_set_hybrid_dequant(resources.decoder, variant != 0), "Select short-plane unpacking");
		check(pyrowave_bench_set_native_idwt(resources.decoder, variant != 0), "Select balanced reconstruction");
		check(pyrowave_bench_set_batched_dequant(resources.decoder, variant != 0), "Select band batching");
	}
	else if (options.compare == "native-batched")
	{
		check(pyrowave_bench_set_native_idwt(resources.decoder, variant != 0), "Select balanced reconstruction");
		check(pyrowave_bench_set_batched_dequant(resources.decoder, variant != 0), "Select band batching");
	}
	else if (options.compare == "render") resources.render_wait = variant == 0;
	else if (options.compare == "render-fused")
	{
		resources.render_fused = variant != 0;
		check(pyrowave_bench_set_rgb_output(resources.decoder,
		      variant ? (__bridge void *)resources.render_output[variant] : nullptr), "Select fused RGB output");
	}
	else if (options.compare == "render-optimized")
	{
		resources.render_wait = variant == 0;
		// Paced 420 fusion has not established a gain. Keep fragment conversion
		// there; 444 reconstructs all three components directly into RGB.
		resources.render_fused = variant != 0 && !options.chroma_420;
		check(pyrowave_bench_set_native_idwt(resources.decoder, variant != 0), "Select balanced reconstruction");
		check(pyrowave_bench_set_batched_dequant(resources.decoder, variant != 0), "Select band batching");
		check(pyrowave_bench_set_rgb_output(resources.decoder, resources.render_fused ?
		      (__bridge void *)resources.render_output[variant] : nullptr), "Select RGB output path");
	}
}

struct DecodeMetrics
{
	Metric gpu, wall, ready, parse, commands, commit, queue, returned, late;
	Metric upload, dequant_cpu, idwt_cpu, dequant_gpu, idwt_gpu;
	size_t count = 0, misses = 0, over_budget = 0;
	double worst_wall = 0.0;

	void reserve(size_t size)
	{
		for (auto *metric : { &gpu, &wall, &ready, &parse, &commands, &commit, &queue, &returned, &late,
		                     &upload, &dequant_cpu, &idwt_cpu, &dequant_gpu, &idwt_gpu })
			metric->values.reserve(size);
	}

	void add(const Sample &sample, double lateness, bool missed, const Options &options)
	{
		count++;
		gpu.add(sample.decode_gpu); wall.add(sample.decode_wall);
		ready.add(sample.packet_to_gpu);
		parse.add(sample.parse_cpu); commands.add(sample.encode_commands_cpu); commit.add(sample.commit_cpu);
		queue.add(sample.commit_to_gpu); returned.add(sample.gpu_to_return); late.add(lateness, true);
		worst_wall = std::max(worst_wall, sample.decode_wall);
		if (options.fps)
		{
			misses += missed;
			over_budget += sample.decode_wall > 1000.0 / options.fps;
		}
		if (options.profile)
		{
			upload.add(sample.stages.upload_cpu_ms);
			dequant_cpu.add(sample.stages.dequant_encode_cpu_ms);
			idwt_cpu.add(sample.stages.idwt_encode_cpu_ms);
			dequant_gpu.add(sample.stages.dequant_gpu_ms); idwt_gpu.add(sample.stages.idwt_gpu_ms);
		}
	}

	void report(const char *label, const Options &options) const
	{
		std::printf("\n%s (%zu samples)\n", label, count);
		std::printf("%-24s %10s %10s %10s %10s %11s\n", "Time (ms)", "avg", "p50", "p95", "p99", "equiv FPS");
		gpu.report(options.render == "none" ? "Decode GPU command" : "Decode + RGB GPU", count);
		ready.report(options.render == "none" ? "Packet to YUV GPU end" : "Packet to RGB GPU end", count);
		wall.report(options.render == "none" ? "Decode wall + wait" : "Packet to RGB + wait", count);
		parse.report("Parse CPU", count); commands.report("Encode commands CPU", count);
		commit.report("Commit CPU", count); queue.report("Commit to GPU start", count);
		returned.report("GPU end to CPU return", count);
		if (options.fps) late.report("Scheduled start lateness", count);
		if (options.profile)
		{
			upload.report("Upload CPU", count);
			dequant_cpu.report("Dequant encoding CPU", count); idwt_cpu.report("IDWT encoding CPU", count);
			dequant_gpu.report("Dequant GPU pass", count); idwt_gpu.report("IDWT GPU pass", count);
		}
		if (options.fps)
			std::printf("%zu fps deadline misses: %zu/%zu; decode over budget: %zu/%zu; worst decode %.3f ms\n",
			            options.fps, misses, count, over_budget, count, worst_wall);
	}
};

struct RecordedSample
{
	size_t index, variant;
	double lateness;
	bool missed;
	Sample sample;
};

void write_samples(const std::string &path, const std::vector<RecordedSample> &records)
{
	if (path.empty()) return;
	using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
	File file(std::fopen(path.c_str(), "wb"), std::fclose);
	if (!file) throw std::runtime_error("Cannot create sample CSV: " + path);
	std::fprintf(file.get(), "index,variant,decode_gpu_ms,decode_wall_ms,parse_cpu_ms,commands_cpu_ms,commit_cpu_ms,commit_to_gpu_ms,gpu_to_return_ms,start_late_ms,deadline_missed,upload_cpu_ms,dequant_cpu_ms,idwt_cpu_ms,dequant_gpu_ms,idwt_gpu_ms,packet_to_gpu_ms\n");
	for (const auto &record : records)
	{
		const auto &s = record.sample;
		std::fprintf(file.get(), "%zu,%zu,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
		             record.index, record.variant, s.decode_gpu, s.decode_wall, s.parse_cpu, s.encode_commands_cpu,
		             s.commit_cpu, s.commit_to_gpu, s.gpu_to_return, record.lateness, int(record.missed),
		             s.stages.upload_cpu_ms, s.stages.dequant_encode_cpu_ms, s.stages.idwt_encode_cpu_ms,
		             s.stages.dequant_gpu_ms, s.stages.idwt_gpu_ms, s.packet_to_gpu);
	}
	if (std::fflush(file.get()) || std::ferror(file.get()))
		throw std::runtime_error("Failed to write sample CSV");
}

int run(int argc, char **argv)
{
	Options options;
	if (!parse_options(argc, argv, options))
		return EXIT_SUCCESS;
	Frame frame;
	if (options.input.empty())
	{
		validate_dimensions(options);
		frame.allocate(options);
		make_synthetic(frame);
	}
	else
		read_first_frame(options, frame);
	Resources resources;
	if (options.worker_qos >= 0 && pthread_set_qos_class_self_np(
			options.worker_qos ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_DEFAULT, 0) != 0)
		throw std::runtime_error("Failed to set worker QoS");
	setup(resources, options, frame);
	pyrowave_bench_encode_diagnostics fixture_diagnostics = {};
	if (options.profile)
		check(pyrowave_bench_get_encode_diagnostics(resources.encoder, &fixture_diagnostics), "Read fixture scratch counters");
	if (options.batched_dequant)
		check(pyrowave_bench_set_batched_dequant(resources.decoder, true), "Enable batched dequant");
	if (options.reduced_idwt_barriers)
		check(pyrowave_bench_set_reduced_idwt_barriers(resources.decoder, true), "Enable reduced IDWT barriers");
	if (options.native_dequant) check(pyrowave_bench_set_native_dequant(resources.decoder, true), "Enable native dequant");
	if (options.hybrid_dequant) check(pyrowave_bench_set_hybrid_dequant(resources.decoder, true), "Enable hybrid dequant");
	if (options.native_idwt) check(pyrowave_bench_set_native_idwt(resources.decoder, true), "Enable native IDWT");
	if (options.fused_idwt) check(pyrowave_bench_set_fused_idwt(resources.decoder, true), "Enable fused IDWT");
	if (options.profile)
		check(pyrowave_bench_set_decode_profiling(resources.decoder, true), "Enable decode profiling");
	if (options.decode_only)
	{
		packetize_frame(resources);
		// Only the cached compressed frame is used below. Destroying the encoder
		// makes it impossible to submit encode work during decode measurements.
		pyrowave_encoder_destroy(resources.encoder);
		resources.encoder = nullptr;
	}
	// The decoder rotates through four lazily allocated upload slots. Prime all
	// of them and the packet parser outside timing, even when --warmup is zero.
	for (int i = 0; i < 4; i++)
	{
		@autoreleasepool
		{
			if (!options.compare.empty()) select_variant(resources, options, i == 1 || i == 2);
			run_sample(resources, options);
		}
	}
	uint32_t major = 0, minor = 0, patch = 0;
	pyrowave_get_api_version(&major, &minor, &patch);
	const char *serial = std::getenv("PYROWAVE_BENCH_SERIAL");
	std::printf("PyroWave Metal %u.%u.%u on %s\n", major, minor, patch, resources.metal_device.name.UTF8String);
	std::printf("Frame: %dx%d  8-bit YUV%s  precision=%d\n",
	            options.width, options.height, options.chroma_420 ? "420" : "444", options.precision);
	qos_class_t current_qos = QOS_CLASS_UNSPECIFIED;
	int relative_priority = 0;
	pthread_get_qos_class_np(pthread_self(), &current_qos, &relative_priority);
	std::printf("Output storage: %s; dequant: %s; IDWT barriers: %s; GPU pass counters: %s; worker QoS: %u\n",
	            options.private_output ? "private" : "shared", options.batched_dequant ? "batched" : "original",
	            options.reduced_idwt_barriers ? "reduced" : "original",
	            options.profile ? "enabled (diagnostic timing)" : "disabled", unsigned(current_qos));
	std::printf("Reconstruction: %s; unpacking: %s; two-level fusion: %s; offscreen RGB: %s.\n",
	            options.native_idwt ? "balanced" : "original", options.hybrid_dequant ? "hybrid" : options.native_dequant ? "packed" : "original",
	            options.fused_idwt ? "enabled" : "disabled", options.render.c_str());
	if (!options.compare.empty())
	{
		std::printf("A/B: %s; %zu measured frames PER variant; alternating ABBA frame order on the same bitstream.\n",
		            options.compare.c_str(), options.frames);
		if (options.compare == "output")
			std::printf("Baseline: %s output; candidate: %s output.\n",
			            options.private_output ? "private" : "shared", options.private_output ? "shared" : "private");
		else if (options.compare == "render")
			std::printf("Baseline: CPU wait then separate RGB render; candidate: decode and RGB render in one command buffer.\n");
		else if (options.compare == "render-fused")
			std::printf("Baseline: decode + fragment RGB; candidate: fused final reconstruction + RGB.\n");
		else if (options.compare == "render-optimized")
			std::printf("Baseline: original decode, CPU wait, fragment RGB; candidate: balanced reconstruction, band batching, one command buffer%s.\n",
			            options.chroma_420 ? ", fragment RGB" : ", fused final reconstruction + RGB");
		else if (options.compare == "native-batched")
			std::printf("Baseline: original decode; candidate: balanced reconstruction and canonical band batching.\n");
		else
			std::printf("Baseline: original shader; candidate: %s.\n",
			            options.compare.c_str());
	}
	if (options.decode_only)
	{
		std::printf("Mode: decode only; one compressed frame cached in setup; encoder destroyed before timing.\n");
		uint64_t checksum = 14695981039346656037ull;
		for (size_t i = 0; i < resources.packet_count; i++)
		{
			const auto &packet = resources.packets[i];
			for (size_t j = 0; j < packet.size; j++)
				checksum = (checksum ^ resources.packet_data[packet.offset + j]) * 1099511628211ull;
		}
		std::printf("Cached packets: %zu; bitstream FNV-1a64 %016llx\n", resources.packet_count,
		            static_cast<unsigned long long>(checksum));
	}
	else
		std::printf("Mode: roundtrip; input=%s; encode dispatch=%s\n",
	            options.gpu_input ? "gpu (resident IOSurfaces)" : "cpu (upload each iteration)",
	            serial && serial[0] == '1' ? "serial (PYROWAVE_BENCH_SERIAL)" : "concurrent");
	std::printf("Source: %s\n", options.input.empty() ? "deterministic mixed synthetic frame" : options.input.c_str());
	std::printf("Budget: %zu bytes (%zu after 4-byte alignment); warmup: %zu; measured frames: %zu\n",
	            options.bytes, options.bytes & ~size_t(3), options.warmup, options.frames);
	if (options.profile)
		std::printf("Fixture coefficient scratch: %u / %llu bytes used; compressed scratch: %llu / %llu bytes used (setup readback).\n",
		            fixture_diagnostics.coefficient_payload_bytes,
		            static_cast<unsigned long long>(fixture_diagnostics.coefficient_payload_capacity_bytes),
		            static_cast<unsigned long long>(fixture_diagnostics.bitstream_payload_words) * 4,
		            static_cast<unsigned long long>(fixture_diagnostics.bitstream_capacity_bytes));
	std::printf("One frame at a time; setup (including 4 priming %s), disk I/O and final output readback are excluded.\n",
	            options.decode_only ? "decodes" : "roundtrips");
	if (options.fps)
		std::printf("Cadence: %zu frames/s; %.3f ms budget per frame (pacing wait excluded from stage timings).\n",
		            options.fps, 1000.0 / double(options.fps));
	std::fflush(stdout);
	Metric encode_gpu, encode_wall, packetize, roundtrip;
	Metric *metrics[] = { &encode_gpu, &encode_wall, &packetize, &roundtrip };
	for (auto *metric : metrics)
		metric->values.reserve(options.frames);
	DecodeMetrics decode_metrics[2];
	for (auto &metric : decode_metrics) metric.reserve(options.frames);
	const size_t variant_count = options.compare.empty() ? 1 : 2;
	const size_t warmup_iterations = options.warmup * variant_count;
	const size_t measured_iterations = options.frames * variant_count;
	std::vector<RecordedSample> records;
	if (!options.samples_path.empty()) records.reserve(measured_iterations);
	double payload_sum = 0.0;
	size_t payload_min = SIZE_MAX, payload_max = 0;
	size_t deadline_misses = 0, processing_over_budget = 0;
	double max_start_lateness = 0.0, max_roundtrip = 0.0;
	const auto pacing_start = Clock::now();
	for (size_t i = 0; i < warmup_iterations + measured_iterations; i++)
	{
		const size_t variant = variant_count == 2 && (i % 4 == 1 || i % 4 == 2) ? 1 : 0;
		if (variant_count == 2) select_variant(resources, options, variant);
		auto scheduled = pacing_start;
		auto deadline = pacing_start;
		if (options.fps)
		{
			scheduled += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(double(i) / double(options.fps)));
			deadline += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(double(i + 1) / double(options.fps)));
			std::this_thread::sleep_until(scheduled);
		}
		const auto actual_start = Clock::now();
		@autoreleasepool
		{
			const Sample sample = run_sample(resources, options);
			const auto completed = sample.completed;
			if (i >= warmup_iterations)
			{
				const double lateness = options.fps ? std::max(0.0, milliseconds(scheduled, actual_start)) : 0.0;
				decode_metrics[variant].add(sample, lateness, options.fps && completed > deadline, options);
				if (!options.samples_path.empty())
					records.push_back({ i - warmup_iterations, variant, lateness, options.fps && completed > deadline, sample });
				max_roundtrip = std::max(max_roundtrip, sample.roundtrip);
				if (options.fps)
				{
					deadline_misses += completed > deadline;
					processing_over_budget += sample.roundtrip > 1000.0 / double(options.fps);
					max_start_lateness = std::max(max_start_lateness, milliseconds(scheduled, actual_start));
				}
				if (!options.decode_only)
				{
					encode_gpu.add(sample.encode_gpu);
					encode_wall.add(sample.encode_wall);
					packetize.add(sample.packetize);
				}
				roundtrip.add(sample.roundtrip);
				payload_sum += double(sample.payload_bytes);
				payload_min = std::min(payload_min, sample.payload_bytes);
				payload_max = std::max(payload_max, sample.payload_bytes);
			}
		}
	}
	if (!options.decode_only)
	{
		std::printf("\n%-24s %10s %10s %10s %10s %11s\n", "Time (ms)", "avg", "p50", "p95", "p99", "equiv FPS");
		encode_gpu.report("Encode GPU command", options.frames);
		encode_wall.report("Encode wall + wait", options.frames);
		packetize.report("Packetize CPU", options.frames);
	}
	if (variant_count == 2)
	{
		decode_metrics[0].report("Baseline", options);
		decode_metrics[1].report("Candidate", options);
	}
	else decode_metrics[0].report("Decode phases", options);
	if (!options.decode_only)
		roundtrip.report("Roundtrip wall", options.frames);
	if (options.fps && !options.decode_only)
	{
		std::printf("%zu fps deadline misses: %zu/%zu (%.2f%%); processing over budget: %zu/%zu\n",
		            options.fps, deadline_misses, options.frames, 100.0 * double(deadline_misses) / double(options.frames),
		            processing_over_budget, options.frames);
		std::printf("Worst %s: %.3f ms; maximum start lateness: %.3f ms\n",
		            options.decode_only ? "decode wall time" : "roundtrip", max_roundtrip, max_start_lateness);
	}
	std::printf("Frame payload: %.1f bytes avg, %zu min, %zu max (including bitstream headers)\n",
	            payload_sum / double(measured_iterations), payload_min, payload_max);
	std::printf("GPU rows measure complete command buffers; wall rows include submission and synchronization.\n");
	std::printf("Packet-to-output starts before parsing cached packets and ends at GPU completion; no network arrival gaps are simulated.\n");
	if (options.render != "none")
		std::printf("Render-wait GPU time sums decode and render commands; its queue row covers the first submission only.\n");
	std::printf("Commit CPU overlaps the commit-to-GPU interval; diagnostic counters may perturb GPU timing.\n");
	std::printf("Equivalent FPS is 1000 / mean stage time, not video playback throughput.\n");
	const auto baseline_pixels = read_output(resources, frame, resources.output);
	validate_output(baseline_pixels, frame);
	if (variant_count == 2)
	{
		if (options.compare != "render-fused" && !(options.compare == "render-optimized" && !options.chroma_420))
		{
			const auto candidate_pixels = read_output(resources, frame, resources.alternate_output);
			if (candidate_pixels != baseline_pixels)
			{
				for (int plane = 0; plane < 3; plane++)
				{
					size_t differing = 0, first = SIZE_MAX; int maximum = 0;
					for (size_t i = 0; i < baseline_pixels[plane].size(); i++)
					{
						int d = std::abs(int(baseline_pixels[plane][i]) - int(candidate_pixels[plane][i]));
						if (d && first == SIZE_MAX) first = i;
						differing += d != 0; maximum = std::max(maximum, d);
					}
					std::fprintf(stderr, "Plane %d: %zu differing pixels, max %d LSB, first at (%zu,%zu)\n",
					             plane, differing, maximum, first == SIZE_MAX ? 0 : first % frame.planes[plane].width,
					             first == SIZE_MAX ? 0 : first / frame.planes[plane].width);
					if (differing && !options.samples_path.empty())
						for (int variant = 0; variant < 2; variant++)
						{
							auto path = options.samples_path + ".plane" + std::to_string(plane) + (variant ? ".candidate.bin" : ".baseline.bin");
							auto file = std::unique_ptr<FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "wb"), std::fclose);
							const auto &pixels = variant ? candidate_pixels[plane] : baseline_pixels[plane];
							if (!file || std::fwrite(pixels.data(), 1, pixels.size(), file.get()) != pixels.size())
								throw std::runtime_error("Failed to write mismatch diagnostic: " + path);
						}
				}
				throw std::runtime_error("A/B decoded output differs: candidate is not bit-identical to baseline");
			}
			std::printf("A/B output: bit-identical on all three planes.\n");
		}
	}
	if (options.render != "none")
	{
		std::vector<uint8_t> rgb(size_t(options.width) * options.height * 4), candidate(rgb.size());
		[resources.render_output[0] getBytes:rgb.data() bytesPerRow:size_t(options.width) * 4
		                        fromRegion:MTLRegionMake2D(0, 0, options.width, options.height) mipmapLevel:0];
		if (variant_count == 2)
		{
			[resources.render_output[1] getBytes:candidate.data() bytesPerRow:size_t(options.width) * 4
			                        fromRegion:MTLRegionMake2D(0, 0, options.width, options.height) mipmapLevel:0];
			if (rgb != candidate)
			{
				size_t differing = 0; int maximum = 0;
				for (size_t i = 0; i < rgb.size(); i++) { int d = std::abs(int(rgb[i]) - int(candidate[i])); differing += d != 0; maximum = std::max(maximum, d); }
				throw std::runtime_error("A/B RGB output differs: " + std::to_string(differing) + " channels, max " + std::to_string(maximum) + " LSB");
			}
			std::printf("A/B offscreen RGB: bit-identical on all channels.\n");
		}
		std::printf("RGB output is offscreen; these timings exclude drawable acquisition and display presentation.\n");
	}
	write_samples(options.samples_path, records);
	return EXIT_SUCCESS;
}
}

int main(int argc, char **argv)
{
	@autoreleasepool
	{
		try
		{
			return run(argc, argv);
		}
		catch (const std::exception &error)
		{
			std::fprintf(stderr, "pyrowave-metal-bench: %s\n", error.what());
			return EXIT_FAILURE;
		}
	}
}
