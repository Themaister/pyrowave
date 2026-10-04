// SPDX-License-Identifier: MIT
#pragma once

// Private benchmark candidate: native, balanced interior apron loading. Keep the
// reviewed lifting arithmetic and the original mirror-edge implementation.
#include "shaders/pyrowave_msl.h"
#include <cstdint>
#include <string>

namespace PyroWave
{
namespace NativeIdwtDetail
{
inline std::string build_source(int precision)
{
	if (precision < 0 || precision > 2)
		return {};
	const char *canonical = precision == 0 ? idwt_fp16_msl_source :
	                        precision == 1 ? idwt_fp16_storage_msl_source : idwt_msl_source;
	static const uint64_t reviewed[] = {
		0xc81ec6deb0807faeull, 0x06519ddf727250c7ull, 0x948127a862a5467bull
	};
	uint64_t fingerprint = 14695981039346656037ull;
	for (auto *p = reinterpret_cast<const unsigned char *>(canonical); *p; p++)
		fingerprint = (fingerprint ^ *p) * 1099511628211ull;
	if (fingerprint != reviewed[precision])
		return {};

	std::string source = canonical;
	const std::string original_name = "void load_image_with_apron(";
	const size_t original = source.find(original_name);
	if (original == std::string::npos || source.find(original_name, original + 1) != std::string::npos)
		return {};
	source.replace(original, original_name.size(), "void load_image_with_apron_canonical(");
	const std::string kernel_name = "kernel void pyrowave_idwt(";
	const size_t kernel = source.find(kernel_name);
	if (kernel == std::string::npos || source.find(kernel_name, kernel + 1) != std::string::npos)
		return {};

	std::string loader = precision == 2 ? "#define PW_NATIVE_PAIR float2\n" : "#define PW_NATIVE_PAIR half2\n";
	loader += precision == 0 ? "#define PW_NATIVE_QUAD half4\n" : "#define PW_NATIVE_QUAD float4\n";
	loader += R"PW_NATIVE_MSL(
// The existing host dispatch must remain exactly 64 threads. The original
// loader covers an 8x8 core, 2x10 right apron, and 8x2 bottom apron: 100 tiles.
// Its first 16 threads load three tiles each. Here the same tiles are assigned
// in two rounds, at most two per thread, while preserving the core's swizzle.
static inline __attribute__((always_inline))
void load_image_with_apron(threadgroup spvUnsafeArray<spvUnsafeArray<PW_NATIVE_PAIR, 41>, 20>& shared_block,
                          constant Registers& registers, thread uint3& gl_WorkGroupID,
                          thread uint& local_index, texture2d_array<float> uTexture,
                          sampler uTextureSmplr)
{
    int2 base = int2(gl_WorkGroupID.xy) * 16 - 2;
    // Uniform across the threadgroup. Edge/partial tiles use the exact original
    // band-dependent mirror coordinates and their original barrier.
    if (any(base < int2(0)) || any(base + int2(20) > registers.resolution))
    {
        load_image_with_apron_canonical(shared_block, registers, gl_WorkGroupID,
                                       local_index, uTexture, uTextureSmplr);
        return;
    }

    for (uint tile = local_index; tile < 100u; tile += 64u)
    {
        int2 local;
        if (tile < 64u)
        {
            uint core = tile;
            local = 2 * unswizzle8x8(core);
        }
        else if (tile < 84u)
        {
            uint right = tile - 64u;
            local = int2(16 + 2 * int(right & 1u), 2 * int(right >> 1u));
        }
        else
        {
            uint bottom = tile - 84u;
            local = int2(2 * int(bottom >> 1u), 16 + 2 * int(bottom & 1u));
        }

        // Interior coordinates are in [0, resolution-2]. Therefore neither
        // generate_mirror_uv adjustment is active for any band: all four use
        // precisely (base + local + 1) * inv_resolution, with transposed axes.
        // Reuse that coordinate and retain gather/swizzle, avoiding four scalar
        // texture reads per gather and preserving the sampler's exact behavior.
        float2 uv = (float2(base + local + 1) * registers.inv_resolution).yx;
        PW_NATIVE_QUAD a = PW_NATIVE_QUAD(uTexture.gather(uTextureSmplr, uv, 0u, int2(0), component::x)).wxzy;
        PW_NATIVE_QUAD b = PW_NATIVE_QUAD(uTexture.gather(uTextureSmplr, uv, 2u, int2(0), component::x)).wxzy;
        PW_NATIVE_QUAD c = PW_NATIVE_QUAD(uTexture.gather(uTextureSmplr, uv, 1u, int2(0), component::x)).wxzy;
        PW_NATIVE_QUAD d = PW_NATIVE_QUAD(uTexture.gather(uTextureSmplr, uv, 3u, int2(0), component::x)).wxzy;
        write_shared_4x4(local, a, b, c, d, shared_block);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#undef PW_NATIVE_PAIR
#undef PW_NATIVE_QUAD

)PW_NATIVE_MSL";
	source.insert(kernel, loader);
	return source;
}
}

// Empty means an invalid precision or changed canonical source, and the caller
// must reject the experiment. Valid results have stable lifetime and retain the
// pyrowave_idwt entry point, Registers, all resources, and function constant 0.
inline const std::string &native_idwt_source(int precision)
{
	static const std::string sources[] = {
		NativeIdwtDetail::build_source(0), NativeIdwtDetail::build_source(1), NativeIdwtDetail::build_source(2)
	};
	static const std::string empty;
	return precision >= 0 && precision <= 2 ? sources[precision] : empty;
}
}
