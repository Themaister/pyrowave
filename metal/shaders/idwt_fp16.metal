#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct Registers
{
    int2 resolution;
    float2 inv_resolution;
};

constant bool DCShift_tmp [[function_constant(0)]];
constant bool DCShift = is_function_constant_defined(DCShift_tmp) ? DCShift_tmp : false;

static inline __attribute__((always_inline))
int2 unswizzle8x8(thread const uint& index)
{
    uint y = extract_bits(index, uint(0), uint(1));
    uint x = extract_bits(index, uint(1), uint(2));
    y |= (extract_bits(index, uint(3), uint(2)) << uint(1));
    x |= (extract_bits(index, uint(5), uint(1)) << uint(2));
    return int2(int(x), int(y));
}

static inline __attribute__((always_inline))
bool2 band(thread const bool2& a, thread const bool2& b)
{
    return bool2(a.x && b.x, a.y && b.y);
}

static inline __attribute__((always_inline))
float2 generate_mirror_uv(thread int2& coord, thread const bool& even_x, thread const bool& even_y, constant Registers& _150)
{
    bool2 param = bool2(even_x, even_y);
    bool2 param_1 = coord < int2(0);
    coord -= int2(band(param, param_1));
    coord += int2(1);
    bool2 param_2 = bool2(!even_x, !even_y);
    bool2 param_3 = coord >= _150.resolution;
    coord += int2(band(param_2, param_3));
    float2 uv = float2(coord) * _150.inv_resolution;
    return uv.yx;
}

static inline __attribute__((always_inline))
void store_shared(thread const uint& y, thread const uint& x, thread const half2& v, threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20>& shared_block)
{
    shared_block[y][x] = v;
}

static inline __attribute__((always_inline))
void write_shared_4x4(thread const int2& coord, thread const half4& texels0, thread const half4& texels1, thread const half4& texels2, thread const half4& texels3, threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20>& shared_block)
{
    uint param = uint(coord.y + 0);
    uint param_1 = uint((2 * coord.x) + 0);
    half2 param_2 = half2(texels0.x, texels2.x);
    store_shared(param, param_1, param_2, shared_block);
    uint param_3 = uint(coord.y + 0);
    uint param_4 = uint((2 * coord.x) + 1);
    half2 param_5 = half2(texels1.x, texels3.x);
    store_shared(param_3, param_4, param_5, shared_block);
    uint param_6 = uint(coord.y + 0);
    uint param_7 = uint((2 * coord.x) + 2);
    half2 param_8 = half2(texels0.y, texels2.y);
    store_shared(param_6, param_7, param_8, shared_block);
    uint param_9 = uint(coord.y + 0);
    uint param_10 = uint((2 * coord.x) + 3);
    half2 param_11 = half2(texels1.y, texels3.y);
    store_shared(param_9, param_10, param_11, shared_block);
    uint param_12 = uint(coord.y + 1);
    uint param_13 = uint((2 * coord.x) + 0);
    half2 param_14 = half2(texels0.z, texels2.z);
    store_shared(param_12, param_13, param_14, shared_block);
    uint param_15 = uint(coord.y + 1);
    uint param_16 = uint((2 * coord.x) + 1);
    half2 param_17 = half2(texels1.z, texels3.z);
    store_shared(param_15, param_16, param_17, shared_block);
    uint param_18 = uint(coord.y + 1);
    uint param_19 = uint((2 * coord.x) + 2);
    half2 param_20 = half2(texels0.w, texels2.w);
    store_shared(param_18, param_19, param_20, shared_block);
    uint param_21 = uint(coord.y + 1);
    uint param_22 = uint((2 * coord.x) + 3);
    half2 param_23 = half2(texels1.w, texels3.w);
    store_shared(param_21, param_22, param_23, shared_block);
}

static inline __attribute__((always_inline))
void load_image_with_apron(threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20>& shared_block, constant Registers& _150, thread uint3& gl_WorkGroupID, thread uint& local_index, texture2d_array<float> uTexture, sampler uTextureSmplr)
{
    int2 base_coord = (int2(gl_WorkGroupID.xy) * int2(16)) - int2(2);
    uint param = local_index;
    int2 local_coord0 = int2(2) * unswizzle8x8(param);
    int2 coord0 = base_coord + local_coord0;
    int2 param_1 = coord0;
    bool param_2 = true;
    bool param_3 = true;
    float2 _356 = generate_mirror_uv(param_1, param_2, param_3, _150);
    float3 _361 = float3(_356, 0.0);
    half4 texels0 = half4(uTexture.gather(uTextureSmplr, _361.xy, uint(rint(_361.z)), int2(0), component::x)).wxzy;
    int2 param_4 = coord0;
    bool param_5 = false;
    bool param_6 = true;
    float2 _373 = generate_mirror_uv(param_4, param_5, param_6, _150);
    float3 _377 = float3(_373, 2.0);
    half4 texels1 = half4(uTexture.gather(uTextureSmplr, _377.xy, uint(rint(_377.z)), int2(0), component::x)).wxzy;
    int2 param_7 = coord0;
    bool param_8 = true;
    bool param_9 = false;
    float2 _387 = generate_mirror_uv(param_7, param_8, param_9, _150);
    float3 _391 = float3(_387, 1.0);
    half4 texels2 = half4(uTexture.gather(uTextureSmplr, _391.xy, uint(rint(_391.z)), int2(0), component::x)).wxzy;
    int2 param_10 = coord0;
    bool param_11 = false;
    bool param_12 = false;
    float2 _401 = generate_mirror_uv(param_10, param_11, param_12, _150);
    float3 _405 = float3(_401, 3.0);
    half4 texels3 = half4(uTexture.gather(uTextureSmplr, _405.xy, uint(rint(_405.z)), int2(0), component::x)).wxzy;
    int2 param_13 = local_coord0;
    half4 param_14 = texels0;
    half4 param_15 = texels1;
    half4 param_16 = texels2;
    half4 param_17 = texels3;
    write_shared_4x4(param_13, param_14, param_15, param_16, param_17, shared_block);
    int2 local_coord_horiz = int2(int(16u + (2u * (local_index % 2u))), int(2u * (local_index / 2u)));
    if (local_coord_horiz.y < 20)
    {
        int2 param_18 = base_coord + local_coord_horiz;
        bool param_19 = true;
        bool param_20 = true;
        float2 _445 = generate_mirror_uv(param_18, param_19, param_20, _150);
        float3 _448 = float3(_445, 0.0);
        texels0 = half4(uTexture.gather(uTextureSmplr, _448.xy, uint(rint(_448.z)), int2(0), component::x)).wxzy;
        int2 param_21 = base_coord + local_coord_horiz;
        bool param_22 = false;
        bool param_23 = true;
        float2 _459 = generate_mirror_uv(param_21, param_22, param_23, _150);
        float3 _462 = float3(_459, 2.0);
        texels1 = half4(uTexture.gather(uTextureSmplr, _462.xy, uint(rint(_462.z)), int2(0), component::x)).wxzy;
        int2 param_24 = base_coord + local_coord_horiz;
        bool param_25 = true;
        bool param_26 = false;
        float2 _473 = generate_mirror_uv(param_24, param_25, param_26, _150);
        float3 _476 = float3(_473, 1.0);
        texels2 = half4(uTexture.gather(uTextureSmplr, _476.xy, uint(rint(_476.z)), int2(0), component::x)).wxzy;
        int2 param_27 = base_coord + local_coord_horiz;
        bool param_28 = false;
        bool param_29 = false;
        float2 _487 = generate_mirror_uv(param_27, param_28, param_29, _150);
        float3 _490 = float3(_487, 3.0);
        texels3 = half4(uTexture.gather(uTextureSmplr, _490.xy, uint(rint(_490.z)), int2(0), component::x)).wxzy;
        int2 param_30 = local_coord_horiz;
        half4 param_31 = texels0;
        half4 param_32 = texels1;
        half4 param_33 = texels2;
        half4 param_34 = texels3;
        write_shared_4x4(param_30, param_31, param_32, param_33, param_34, shared_block);
    }
    int2 local_coord_vert = local_coord_horiz.yx;
    if (local_coord_vert.x < 16)
    {
        int2 param_35 = base_coord + local_coord_vert;
        bool param_36 = true;
        bool param_37 = true;
        float2 _520 = generate_mirror_uv(param_35, param_36, param_37, _150);
        float3 _523 = float3(_520, 0.0);
        texels0 = half4(uTexture.gather(uTextureSmplr, _523.xy, uint(rint(_523.z)), int2(0), component::x)).wxzy;
        int2 param_38 = base_coord + local_coord_vert;
        bool param_39 = false;
        bool param_40 = true;
        float2 _534 = generate_mirror_uv(param_38, param_39, param_40, _150);
        float3 _537 = float3(_534, 2.0);
        texels1 = half4(uTexture.gather(uTextureSmplr, _537.xy, uint(rint(_537.z)), int2(0), component::x)).wxzy;
        int2 param_41 = base_coord + local_coord_vert;
        bool param_42 = true;
        bool param_43 = false;
        float2 _548 = generate_mirror_uv(param_41, param_42, param_43, _150);
        float3 _551 = float3(_548, 1.0);
        texels2 = half4(uTexture.gather(uTextureSmplr, _551.xy, uint(rint(_551.z)), int2(0), component::x)).wxzy;
        int2 param_44 = base_coord + local_coord_vert;
        bool param_45 = false;
        bool param_46 = false;
        float2 _562 = generate_mirror_uv(param_44, param_45, param_46, _150);
        float3 _565 = float3(_562, 3.0);
        texels3 = half4(uTexture.gather(uTextureSmplr, _565.xy, uint(rint(_565.z)), int2(0), component::x)).wxzy;
        int2 param_47 = local_coord_vert;
        half4 param_48 = texels0;
        half4 param_49 = texels1;
        half4 param_50 = texels2;
        half4 param_51 = texels3;
        write_shared_4x4(param_47, param_48, param_49, param_50, param_51, shared_block);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline __attribute__((always_inline))
half2 load_shared(thread const uint& y, thread const uint& x, threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20>& shared_block)
{
    return shared_block[y][x];
}

static inline __attribute__((always_inline))
void inverse_transform8x2(threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20>& shared_block, thread uint& local_index)
{
    int2 local_coord = int2(int(8u * (local_index % 4u)), int(local_index / 4u));
    spvUnsafeArray<half2, 16> values;
    for (int i = 0; i < 16; i += 2)
    {
        uint param = uint(local_coord.y);
        uint param_1 = uint((local_coord.x + i) + 0);
        half2 v0 = load_shared(param, param_1, shared_block);
        uint param_2 = uint(local_coord.y);
        uint param_3 = uint((local_coord.x + i) + 1);
        half2 v1 = load_shared(param_2, param_3, shared_block);
        values[i + 0] = v0 * half(1.2294921875);
        values[i + 1] = v1 * half(0.8125);
    }
    for (int i_1 = 2; i_1 < 15; i_1 += 2)
    {
        values[i_1] -= ((values[i_1 - 1] + values[i_1 + 1]) * half(0.443359375));
    }
    for (int i_2 = 3; i_2 < 14; i_2 += 2)
    {
        values[i_2] -= ((values[i_2 - 1] + values[i_2 + 1]) * half(0.8828125));
    }
    for (int i_3 = 4; i_3 < 13; i_3 += 2)
    {
        values[i_3] -= ((values[i_3 - 1] + values[i_3 + 1]) * half(-0.052978515625));
    }
    for (int i_4 = 5; i_4 < 12; i_4 += 2)
    {
        values[i_4] -= ((values[i_4 - 1] + values[i_4 + 1]) * half(-1.5859375));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i_5 = 2; i_5 < 6; i_5++)
    {
        half2 a = values[(2 * i_5) + 0];
        half2 b = values[(2 * i_5) + 1];
        half2 t0 = half2(a.x, b.x);
        half2 t1 = half2(a.y, b.y);
        int y_coord = (local_coord.x >> 1) + (i_5 - 2);
        uint param_4 = uint(y_coord);
        uint param_5 = uint((2 * local_coord.y) + 0);
        half2 param_6 = t0;
        store_shared(param_4, param_5, param_6, shared_block);
        uint param_7 = uint(y_coord);
        uint param_8 = uint((2 * local_coord.y) + 1);
        half2 param_9 = t1;
        store_shared(param_7, param_8, param_9, shared_block);
    }
}

static inline __attribute__((always_inline))
void inverse_transform4x2(thread const bool& active_lane, thread const int& y_offset, threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20>& shared_block, thread uint& local_index)
{
    int2 local_coord = int2(int(4u * (local_index % 8u)), int((local_index / 8u) + uint(y_offset)));
    spvUnsafeArray<half2, 12> values;
    if (active_lane)
    {
        for (int i = 0; i < 12; i += 2)
        {
            uint param = uint(local_coord.y);
            uint param_1 = uint((local_coord.x + i) + 0);
            half2 v0 = load_shared(param, param_1, shared_block);
            uint param_2 = uint(local_coord.y);
            uint param_3 = uint((local_coord.x + i) + 1);
            half2 v1 = load_shared(param_2, param_3, shared_block);
            values[i + 0] = v0 * half(1.2294921875);
            values[i + 1] = v1 * half(0.8125);
        }
        for (int i_1 = 2; i_1 < 11; i_1 += 2)
        {
            values[i_1] -= ((values[i_1 - 1] + values[i_1 + 1]) * half(0.443359375));
        }
        for (int i_2 = 3; i_2 < 10; i_2 += 2)
        {
            values[i_2] -= ((values[i_2 - 1] + values[i_2 + 1]) * half(0.8828125));
        }
        for (int i_3 = 4; i_3 < 9; i_3 += 2)
        {
            values[i_3] -= ((values[i_3 - 1] + values[i_3 + 1]) * half(-0.052978515625));
        }
        for (int i_4 = 5; i_4 < 8; i_4 += 2)
        {
            values[i_4] -= ((values[i_4 - 1] + values[i_4 + 1]) * half(-1.5859375));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active_lane)
    {
        for (int i_5 = 2; i_5 < 4; i_5++)
        {
            half2 a = values[(2 * i_5) + 0];
            half2 b = values[(2 * i_5) + 1];
            half2 t0 = half2(a.x, b.x);
            half2 t1 = half2(a.y, b.y);
            int y_coord = (local_coord.x >> 1) + (i_5 - 2);
            uint param_4 = uint(y_coord);
            uint param_5 = uint((2 * local_coord.y) + 0);
            half2 param_6 = t0;
            store_shared(param_4, param_5, param_6, shared_block);
            uint param_7 = uint(y_coord);
            uint param_8 = uint((2 * local_coord.y) + 1);
            half2 param_9 = t1;
            store_shared(param_7, param_8, param_9, shared_block);
        }
    }
}

kernel void pyrowave_idwt(constant Registers& _150 [[buffer(0)]], texture2d_array<float> uTexture [[texture(0)]], texture2d<float, access::write> uOutput [[texture(1)]], sampler uTextureSmplr [[sampler(0)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]])
{
    threadgroup spvUnsafeArray<spvUnsafeArray<half2, 41>, 20> shared_block;
    uint local_index = gl_LocalInvocationIndex;
    load_image_with_apron(shared_block, _150, gl_WorkGroupID, local_index, uTexture, uTextureSmplr);
    inverse_transform8x2(shared_block, local_index);
    bool param = local_index < 32u;
    int param_1 = 16;
    inverse_transform4x2(param, param_1, shared_block, local_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inverse_transform8x2(shared_block, local_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint param_2 = local_index;
    int2 local_coord = unswizzle8x8(param_2);
    for (int y = local_coord.y; y < 16; y += 8)
    {
        for (int x = local_coord.x; x < 32; x += 8)
        {
            uint param_3 = uint(y);
            uint param_4 = uint(x);
            half2 v = load_shared(param_3, param_4, shared_block);
            if (DCShift)
            {
                v += half2(half(0.5));
            }
            uOutput.write(float4(v.xxxx), uint2((int2((2 * y) + 0, x) + (int2(32) * int2(gl_WorkGroupID.yx)))));
            uOutput.write(float4(v.yyyy), uint2((int2((2 * y) + 1, x) + (int2(32) * int2(gl_WorkGroupID.yx)))));
        }
    }
}

