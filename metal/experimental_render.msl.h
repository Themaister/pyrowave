// SPDX-License-Identifier: MIT
// Private offscreen render experiments. Full-range BT.709, nearest chroma.
#pragma once

#include "shaders/pyrowave_msl.h"
#include <cstdint>
#include <string>

namespace PyroWave
{
static const char bench_rgb_math_msl[] = R"MSL(
static inline float4 bench_yuv_rgb(float y, float cb, float cr)
{
    cb -= 0.5f;
    cr -= 0.5f;
    return float4(y + 1.5748f * cr,
                  y - 0.187324f * cb - 0.468124f * cr,
                  y + 1.8556f * cb, 1.0f);
}
)MSL";

inline std::string bench_render_source()
{
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n") + bench_rgb_math_msl + R"MSL(
struct BenchVertex { float4 position [[position]]; };
vertex BenchVertex bench_fullscreen(uint id [[vertex_id]])
{
    const float2 positions[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    return { float4(positions[id], 0, 1) };
}
fragment float4 bench_rgb_fragment(BenchVertex v [[stage_in]],
    texture2d<float, access::read> y [[texture(0)]],
    texture2d<float, access::read> cb [[texture(1)]],
    texture2d<float, access::read> cr [[texture(2)]])
{
    uint2 p = uint2(v.position.xy);
    uint2 c = p * uint2(cb.get_width(), cb.get_height()) / uint2(y.get_width(), y.get_height());
    return bench_yuv_rgb(y.read(p).r, cb.read(c).r, cr.read(c).r);
}
)MSL";
}

inline std::string bench_fused_rgb_source(int precision)
{
    if (precision < 0 || precision > 2) return {};
    const char *canonical = precision == 0 ? idwt_fp16_msl_source :
                            precision == 1 ? idwt_fp16_storage_msl_source : idwt_msl_source;
    static const uint64_t reviewed[] = {
        0xc81ec6deb0807faeull, 0x06519ddf727250c7ull, 0x948127a862a5467bull
    };
    uint64_t fingerprint = 14695981039346656037ull;
    for (auto *p = reinterpret_cast<const unsigned char *>(canonical); *p; p++)
        fingerprint = (fingerprint ^ *p) * 1099511628211ull;
    if (fingerprint != reviewed[precision]) return {};
    std::string source(canonical);
    source += bench_rgb_math_msl;
    const std::string storage = precision == 2 ? "float2" : "half2";
    const std::string value = precision == 0 ? "half2" : "float2";
    // Keep the tested R8 checkpoint for half-storage variants. Full FP32 needs
    // the native conversion: manual multiplication/rounding differed at 4K.
    const std::string checkpoint = precision == 2 ?
        "            uint packed = pack_float_to_unorm4x8(float4(float2(v), 0.0f, 0.0f));\n"
        "            bytes[lane * 8 + i++] = uchar2(packed & 255u, (packed >> 8) & 255u);\n" :
        "            bytes[lane * 8 + i++] = uchar2(rint(clamp(float2(v), 0.0f, 1.0f) * 255.0f));\n";
    source += "\nstatic inline void bench_reconstruct_bytes(\n"
              "    threadgroup spvUnsafeArray<spvUnsafeArray<" + storage + ", 41>, 20>& shared_block,\n"
              "    threadgroup uchar2 *bytes, constant Registers& regs,\n"
              "    texture2d_array<float> input, sampler mirror_sampler, uint3 group, uint lane)\n{\n"
              "    load_image_with_apron(shared_block, regs, group, lane, input, mirror_sampler);\n"
              "    inverse_transform8x2(shared_block, lane);\n"
              "    bool active = lane < 32; int offset = 16;\n"
              "    inverse_transform4x2(active, offset, shared_block, lane);\n"
              "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
              "    inverse_transform8x2(shared_block, lane);\n"
              "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
              "    int2 local = unswizzle8x8(lane); uint i = 0;\n"
              "    for (int y = local.y; y < 16; y += 8)\n"
              "        for (int x = local.x; x < 32; x += 8)\n        {\n"
              "            uint sy = uint(y), sx = uint(x);\n"
              "            " + value + " v = load_shared(sy, sx, shared_block);\n"
              "            if (DCShift) v += " + value + "(0.5);\n"
              + checkpoint +
              "        }\n"
              // Every lane must finish reading the tile before the next component loads it.
              "    threadgroup_barrier(mem_flags::mem_threadgroup);\n}\n"
              "kernel void pyrowave_idwt_rgb(constant Registers& regs [[buffer(0)]],\n"
              "    texture2d_array<float> y_wave [[texture(0)]],\n"
              "    texture2d_array<float> cb_wave [[texture(1)]],\n"
              "    texture2d_array<float> cr_wave [[texture(2)]],\n"
              "    texture2d<float, access::write> rgb [[texture(3)]],\n"
              "    sampler mirror_sampler [[sampler(0)]],\n"
              "    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])\n{\n"
              "    threadgroup spvUnsafeArray<spvUnsafeArray<" + storage + ", 41>, 20> shared_block;\n"
              "    threadgroup uchar2 bytes[3 * 64 * 8];\n"
              "    bench_reconstruct_bytes(shared_block, bytes, regs, y_wave, mirror_sampler, group, lane);\n"
              "    bench_reconstruct_bytes(shared_block, bytes + 64 * 8, regs, cb_wave, mirror_sampler, group, lane);\n"
              "    bench_reconstruct_bytes(shared_block, bytes + 2 * 64 * 8, regs, cr_wave, mirror_sampler, group, lane);\n"
              "    int2 local = unswizzle8x8(lane); uint i = 0;\n"
              "    for (int y = local.y; y < 16; y += 8)\n"
              "        for (int x = local.x; x < 32; x += 8)\n        {\n"
              "            float2 yy = float2(bytes[lane * 8 + i]) / 255.0f;\n"
              "            float2 cb = float2(bytes[64 * 8 + lane * 8 + i]) / 255.0f;\n"
              "            float2 cr = float2(bytes[2 * 64 * 8 + lane * 8 + i++]) / 255.0f;\n"
              "            uint2 p = uint2(int2(2 * y, x) + 32 * int2(group.yx));\n"
              "            rgb.write(bench_yuv_rgb(yy.x, cb.x, cr.x), p);\n"
              "            rgb.write(bench_yuv_rgb(yy.y, cb.y, cr.y), p + uint2(1, 0));\n"
              "        }\n}\n"
              "kernel void pyrowave_idwt_rgb420(constant Registers& regs [[buffer(0)]],\n"
              "    texture2d_array<float> y_wave [[texture(0)]],\n"
              "    texture2d<float, access::read> cb_plane [[texture(1)]],\n"
              "    texture2d<float, access::read> cr_plane [[texture(2)]],\n"
              "    texture2d<float, access::write> rgb [[texture(3)]],\n"
              "    sampler mirror_sampler [[sampler(0)]],\n"
              "    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])\n{\n"
              "    threadgroup spvUnsafeArray<spvUnsafeArray<" + storage + ", 41>, 20> shared_block;\n"
              "    threadgroup uchar2 bytes[64 * 8];\n"
              "    bench_reconstruct_bytes(shared_block, bytes, regs, y_wave, mirror_sampler, group, lane);\n"
              "    int2 local = unswizzle8x8(lane); uint i = 0;\n"
              "    for (int y = local.y; y < 16; y += 8)\n"
              "        for (int x = local.x; x < 32; x += 8)\n        {\n"
              "            float2 yy = float2(bytes[lane * 8 + i++]) / 255.0f;\n"
              "            uint2 p = uint2(int2(2 * y, x) + 32 * int2(group.yx));\n"
              "            for (uint j = 0; j < 2; j++)\n            {\n"
              "                uint2 q = p + uint2(j, 0);\n"
              "                if (q.x < rgb.get_width() && q.y < rgb.get_height())\n                {\n"
              "                    uint2 c = q * uint2(cb_plane.get_width(), cb_plane.get_height()) / uint2(rgb.get_width(), rgb.get_height());\n"
              "                    rgb.write(bench_yuv_rgb(yy[j], cb_plane.read(c).r, cr_plane.read(c).r), q);\n"
              "                }\n            }\n        }\n}\n";
    return source;
}
}
