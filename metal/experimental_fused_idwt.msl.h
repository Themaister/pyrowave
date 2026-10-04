// SPDX-License-Identifier: MIT
#pragma once

// Private benchmark prototype: reconstruct two adjacent levels in one dispatch.
// The original lifting helpers are retained exactly, including their arithmetic,
// half-storage rounding and synchronization. The intermediate LL texture is
// replaced with a threadgroup patch. This intentionally simple prototype repeats
// a coarse 32x32 reconstruction per fine tile; measure before adopting it.
#include "experimental_dequant.msl.h"
#include "shaders/pyrowave_msl.h"
#include <stdint.h>

namespace PyroWave
{
inline bool build_fused_idwt_msl(const char *canonical_source, int precision, std::string &source)
{
	static const uint64_t reviewed_sources[] = {
		0xc81ec6deb0807faeull, // FP16 math/storage.
		0x06519ddf727250c7ull, // FP32 math, FP16 storage.
		0x948127a862a5467bull  // FP32 math/storage.
	};
	if (precision < 0 || precision > 2)
		return false;
	uint64_t fingerprint = 14695981039346656037ull;
	for (auto *p = reinterpret_cast<const unsigned char *>(canonical_source); *p; p++)
		fingerprint = (fingerprint ^ *p) * 1099511628211ull;
	if (fingerprint != reviewed_sources[precision])
		return false;

	source = canonical_source;
	const char *register_name = precision == 0 ? "_150" : precision == 1 ? "_152" : "_148";
	const char *math_scalar = precision == 0 ? "half" : "float";
	const char *math_vec2 = precision == 0 ? "half2" : "float2";
	const char *math_vec4 = precision == 0 ? "half4" : "float4";
	const char *storage_scalar = precision == 2 ? "float" : "half";
	const char *storage_vec2 = precision == 2 ? "float2" : "half2";
	const std::string shared_type = std::string("spvUnsafeArray<spvUnsafeArray<") + storage_vec2 + ", 41>, 20>";
	const std::string patch_type = std::string("spvUnsafeArray<") + storage_scalar + ", 1024>";
	const std::string load_begin = "static inline __attribute__((always_inline))\nvoid load_image_with_apron(";
	const size_t load_start = source.find(load_begin);
	const size_t load_end = source.find("static inline __attribute__((always_inline))\n", load_start + load_begin.size());
	if (load_start == std::string::npos || load_end == std::string::npos)
		return false;
	const std::string original_load = source.substr(load_start, load_end - load_start);
	std::string coarse_load = original_load;
	if (!replace_unique_msl(coarse_load, "void load_image_with_apron(", "void load_coarse_with_apron(") ||
	    !replace_unique_msl(coarse_load, "thread uint3& gl_WorkGroupID", "thread const int2& patch_base") ||
	    !replace_unique_msl(coarse_load,
	        "int2 base_coord = (int2(gl_WorkGroupID.xy) * int2(16)) - int2(2);",
	        "int2 base_coord = patch_base.yx / int2(2) - int2(2);"))
		return false;

	std::string fine_load = original_load;
	const std::string extra_arguments = std::string("sampler uTextureSmplr, thread const int2& patch_base, threadgroup ") + patch_type + "& ll_patch)";
	if (!replace_unique_msl(fine_load, "void load_image_with_apron(", "void load_fine_with_apron(") ||
	    !replace_unique_msl(fine_load, "sampler uTextureSmplr)", extra_arguments.c_str()))
		return false;

	// Only the three LL gathers are replaced. Their UVs still pass through the
	// canonical parity-aware mirror adjustment; all high-band gathers are intact.
	size_t search = 0;
	unsigned replaced = 0;
	while ((search = fine_load.find("texels0 = ", search)) != std::string::npos)
	{
		const size_t gather = fine_load.find("uTexture.gather(uTextureSmplr, ", search);
		const size_t semicolon = fine_load.find(';', search);
		if (gather == std::string::npos || gather > semicolon)
			return false;
		const size_t uv_start = gather + std::char_traits<char>::length("uTexture.gather(uTextureSmplr, ");
		const size_t uv_end = fine_load.find(".xy", uv_start);
		if (uv_end == std::string::npos || uv_end > semicolon)
			return false;
		const std::string uv = fine_load.substr(uv_start, uv_end + 3 - uv_start);
		const size_t rhs_start = search + std::char_traits<char>::length("texels0 = ");
		const std::string rhs = "gather_fused_ll(" + uv + ", " + register_name + ", patch_base, ll_patch)";
		fine_load.replace(rhs_start, semicolon - rhs_start, rhs);
		search = rhs_start + rhs.size();
		replaced++;
	}
	if (replaced != 3)
		return false;

	const std::string helpers = std::string(
		"struct FusedRegisters\n{\n    Registers fine;\n    Registers coarse;\n};\n\n"
		"static inline __attribute__((always_inline))\n"
		"int fused_mirror_index(int index, int size)\n"
		"{\n    int p = index % (2 * size);\n    if (p < 0) p += 2 * size;\n    return p < size ? p : 2 * size - 1 - p;\n}\n\n") +
		"static inline __attribute__((always_inline))\n" + math_vec4 + " gather_fused_ll(float2 uv, constant Registers& registers, thread const int2& patch_base, threadgroup " + patch_type + "& ll_patch)\n"
		"{\n"
		"    int2 size = registers.resolution.yx;\n"
		"    int2 coord = int2(floor(uv * float2(size) - 0.5f));\n"
		"    int x0 = fused_mirror_index(coord.x, size.x) - patch_base.x;\n"
		"    int x1 = fused_mirror_index(coord.x + 1, size.x) - patch_base.x;\n"
		"    int y0 = fused_mirror_index(coord.y, size.y) - patch_base.y;\n"
		"    int y1 = fused_mirror_index(coord.y + 1, size.y) - patch_base.y;\n"
		"    // Match texture.gather(...).wxzy after the canonical transpose.\n" +
		"    return " + math_vec4 + "(ll_patch[y0 * 32 + x0], ll_patch[y1 * 32 + x0], ll_patch[y0 * 32 + x1], ll_patch[y1 * 32 + x1]);\n"
		"}\n\n";
	source.replace(load_start, load_end - load_start, helpers + coarse_load + fine_load);
	const size_t kernel_start = source.find("kernel void pyrowave_idwt(");
	if (kernel_start == std::string::npos)
		return false;
	source.resize(kernel_start);
	source +=
		"kernel void pyrowave_idwt_fused(constant FusedRegisters& push [[buffer(0)]], texture2d_array<float> uCoarse [[texture(0)]], texture2d_array<float> uFine [[texture(1)]], texture2d<float, access::write> uOutput [[texture(2)]], sampler uTextureSmplr [[sampler(0)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]])\n"
		"{\n"
		"    threadgroup " + shared_type + " shared_block;\n"
		"    threadgroup " + patch_type + " ll_patch;\n"
		"    uint local_index = gl_LocalInvocationIndex;\n"
		"    int2 fine_size = push.fine.resolution.yx;\n"
		"    int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 4, int2(0), fine_size - 32);\n"
		"    load_coarse_with_apron(shared_block, push.coarse, patch_base, local_index, uCoarse, uTextureSmplr);\n"
		"    inverse_transform8x2(shared_block, local_index);\n"
		"    bool active_lane = local_index < 32u;\n"
		"    int y_offset = 16;\n"
		"    inverse_transform4x2(active_lane, y_offset, shared_block, local_index);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    inverse_transform8x2(shared_block, local_index);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    int2 local_coord = unswizzle8x8(local_index);\n"
		"    for (int y = local_coord.y; y < 16; y += 8)\n"
		"        for (int x = local_coord.x; x < 32; x += 8)\n"
		"        {\n"
		"            uint sy = uint(y), sx = uint(x);\n"
		"            " + math_vec2 + " v = load_shared(sy, sx, shared_block);\n"
		"            ll_patch[x * 32 + 2 * y] = " + storage_scalar + "(v.x);\n"
		"            ll_patch[x * 32 + 2 * y + 1] = " + storage_scalar + "(v.y);\n"
		"        }\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    load_fine_with_apron(shared_block, push.fine, gl_WorkGroupID, local_index, uFine, uTextureSmplr, patch_base, ll_patch);\n"
		"    inverse_transform8x2(shared_block, local_index);\n"
		"    inverse_transform4x2(active_lane, y_offset, shared_block, local_index);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    inverse_transform8x2(shared_block, local_index);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    for (int y = local_coord.y; y < 16; y += 8)\n"
		"        for (int x = local_coord.x; x < 32; x += 8)\n"
		"        {\n"
		"            uint sy = uint(y), sx = uint(x);\n"
		"            " + math_vec2 + " v = load_shared(sy, sx, shared_block);\n"
		"            if (DCShift) v += " + math_vec2 + "(" + math_scalar + "(0.5));\n"
		"            uOutput.write(float4(v.xxxx), uint2(int2(2 * y, x) + int2(32) * int2(gl_WorkGroupID.yx)));\n"
		"            uOutput.write(float4(v.yyyy), uint2(int2(2 * y + 1, x) + int2(32) * int2(gl_WorkGroupID.yx)));\n"
		"        }\n"
		"}\n";
	return true;
}

// A narrower patch reduces duplicate coarse reconstruction. The fine transform
// needs 20x20 LL coefficients; 24x24 is the smallest whole 8-pixel lifting tile
// which covers that footprint. Arithmetic and storage remain canonical.
inline bool build_compact_fused_idwt_msl(const char *canonical_source, int precision, std::string &source)
{
	if (!build_fused_idwt_msl(canonical_source, precision, source))
		return false;
	const size_t coarse_start = source.find("void load_coarse_with_apron(");
	const size_t coarse_end = source.find("static inline __attribute__((always_inline))\n", coarse_start);
	if (coarse_start == std::string::npos || coarse_end == std::string::npos)
		return false;
	std::string coarse_load = source.substr(coarse_start, coarse_end - coarse_start);
	const size_t first_store = coarse_load.find("    write_shared_4x4(");
	const size_t first_store_end = coarse_load.find(';', first_store);
	if (first_store == std::string::npos || first_store_end == std::string::npos)
		return false;
	coarse_load.resize(first_store_end + 1);
	coarse_load += "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n}\n\n";
	source.replace(coarse_start, coarse_end - coarse_start, coarse_load);

	const size_t transform_start = source.find("static inline __attribute__((always_inline))\nvoid inverse_transform8x2(");
	const size_t transform_end = source.find("static inline __attribute__((always_inline))\n", transform_start + 1);
	if (transform_start == std::string::npos || transform_end == std::string::npos)
		return false;
	std::string transform = source.substr(transform_start, transform_end - transform_start);
	if (!replace_unique_msl(transform, "void inverse_transform8x2(", "void inverse_transform_compact_coarse(") ||
	    !replace_unique_msl(transform, "thread uint& local_index)", "thread uint& local_index, thread const bool& active_lane)") ||
	    !replace_unique_msl(transform,
	        "int2 local_coord = int2(int(8u * (local_index % 4u)), int(local_index / 4u));",
	        "int2 local_coord = int2(int(8u * (local_index % 3u)), int(local_index / 3u));") ||
	    !replace_unique_msl(transform, "    for (int i = 0; i < 16; i += 2)", "    if (active_lane)\n    {\n    for (int i = 0; i < 16; i += 2)") ||
	    !replace_unique_msl(transform,
	        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (int i_5 = 2; i_5 < 6; i_5++)",
	        "    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (active_lane)\n    {\n    for (int i_5 = 2; i_5 < 6; i_5++)"))
		return false;
	const size_t final_brace = transform.rfind('}');
	if (final_brace == std::string::npos)
		return false;
	transform.insert(final_brace, "    }\n");
	source.insert(transform_start, transform);

	const char *before =
		"    load_coarse_with_apron(shared_block, push.coarse, patch_base, local_index, uCoarse, uTextureSmplr);\n"
		"    inverse_transform8x2(shared_block, local_index);\n"
		"    bool active_lane = local_index < 32u;\n"
		"    int y_offset = 16;\n"
		"    inverse_transform4x2(active_lane, y_offset, shared_block, local_index);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    inverse_transform8x2(shared_block, local_index);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    int2 local_coord = unswizzle8x8(local_index);\n"
		"    for (int y = local_coord.y; y < 16; y += 8)\n"
		"        for (int x = local_coord.x; x < 32; x += 8)";
	const char *after =
		"    load_coarse_with_apron(shared_block, push.coarse, patch_base, local_index, uCoarse, uTextureSmplr);\n"
		"    bool horizontal_active = local_index < 48u;\n"
		"    inverse_transform_compact_coarse(shared_block, local_index, horizontal_active);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    bool vertical_active = local_index < 36u;\n"
		"    inverse_transform_compact_coarse(shared_block, local_index, vertical_active);\n"
		"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
		"    bool active_lane = local_index < 32u;\n"
		"    int y_offset = 16;\n"
		"    int2 local_coord = unswizzle8x8(local_index);\n"
		"    for (int y = local_coord.y; y < 12; y += 8)\n"
		"        for (int x = local_coord.x; x < 24; x += 8)";
	if (!replace_unique_msl(source, before, after) ||
	    !replace_unique_msl(source,
	        "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 4, int2(0), fine_size - 32);",
	        "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 2, int2(0), fine_size - 24);"))
		return false;
	const char *replacements[][2] = {
		{", 1024>", ", 576>"},
		{"y0 * 32 +", "y0 * 24 +"}, {"y1 * 32 +", "y1 * 24 +"},
		{"x * 32 + 2 * y", "x * 24 + 2 * y"}
	};
	for (const auto &replacement : replacements)
	{
		size_t position = 0;
		while ((position = source.find(replacement[0], position)) != std::string::npos)
		{
			source.replace(position, std::char_traits<char>::length(replacement[0]), replacement[1]);
			position += std::char_traits<char>::length(replacement[1]);
		}
	}
	return true;
}

// Stable cached lifetime for the device's private pipeline initializer. Empty
// source signals unsupported precision or a canonical-source fingerprint change.
inline const std::string &fused_idwt_source(int precision)
{
	static const std::string sources[] = {
		[] { std::string source; if (!build_fused_idwt_msl(idwt_fp16_msl_source, 0, source)) source.clear(); return source; }(),
		[] { std::string source; if (!build_fused_idwt_msl(idwt_fp16_storage_msl_source, 1, source)) source.clear(); return source; }(),
		[] { std::string source; if (!build_fused_idwt_msl(idwt_msl_source, 2, source)) source.clear(); return source; }()
	};
	static const std::string empty;
	return precision >= 0 && precision <= 2 ? sources[precision] : empty;
}

// Same host signature and entry as fused_idwt_source; a separate cache lets the
// benchmark compare the original 32x32 patch and the pruned 24x24 patch.
inline const std::string &compact_fused_idwt_source(int precision)
{
	static const std::string sources[] = {
		[] { std::string source; if (!build_compact_fused_idwt_msl(idwt_fp16_msl_source, 0, source)) source.clear(); return source; }(),
		[] { std::string source; if (!build_compact_fused_idwt_msl(idwt_fp16_storage_msl_source, 1, source)) source.clear(); return source; }(),
		[] { std::string source; if (!build_compact_fused_idwt_msl(idwt_msl_source, 2, source)) source.clear(); return source; }()
	};
	static const std::string empty;
	return precision >= 0 && precision <= 2 ? sources[precision] : empty;
}

// Diagnostic variant: keep the coarse output origin on the original lifting
// helper's 8-pixel segment phase. Algebraically the transform is translation
// invariant at any even origin, but fast-math instruction selection can differ
// between unrolled vector-array positions. This wrapper retains the 32x32 patch
// so the fine 20x20 LL footprint still fits when the origin moves back 8 pixels.
inline const std::string &phase_aligned_fused_idwt_source(int precision)
{
	static const std::string sources[] = {
		[] {
			std::string source = fused_idwt_source(0);
			if (!replace_unique_msl(source,
			    "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 4, int2(0), fine_size - 32);",
			    "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 8, int2(0), fine_size - 32);")) source.clear();
			return source;
		}(),
		[] {
			std::string source = fused_idwt_source(1);
			if (!replace_unique_msl(source,
			    "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 4, int2(0), fine_size - 32);",
			    "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 8, int2(0), fine_size - 32);")) source.clear();
			return source;
		}(),
		[] {
			std::string source = fused_idwt_source(2);
			if (!replace_unique_msl(source,
			    "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 4, int2(0), fine_size - 32);",
			    "int2 patch_base = clamp(int2(gl_WorkGroupID.yx) * 16 - 8, int2(0), fine_size - 32);")) source.clear();
			return source;
		}()
	};
	static const std::string empty;
	return precision >= 0 && precision <= 2 ? sources[precision] : empty;
}
}
