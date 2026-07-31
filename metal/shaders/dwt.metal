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
    int2 aligned_resolution;
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
float2 generate_mirror_uv(thread int2& coord, constant Registers& _104)
{
    coord -= int2(coord < int2(0));
    coord += int2(1);
    int2 end_mirrored_clamp = (int2(2) * _104.aligned_resolution) - _104.resolution;
    int2 past_wrapped_coord = (coord + (int2(2) * (_104.resolution - _104.aligned_resolution))) + int2(1);
    coord = select(min(coord, _104.resolution), past_wrapped_coord, coord >= end_mirrored_clamp);
    return float2(coord) * _104.inv_resolution;
}

static inline __attribute__((always_inline))
void store_shared(thread const uint& y, thread const uint& x, thread const float2& v, threadgroup spvUnsafeArray<spvUnsafeArray<float2, 41>, 20>& shared_block)
{
    shared_block[y][x] = v;
}

static inline __attribute__((always_inline))
void load_image_with_apron(threadgroup spvUnsafeArray<spvUnsafeArray<float2, 41>, 20>& shared_block, constant Registers& _104, thread uint3& gl_WorkGroupID, thread uint& local_index, texture2d<float> uTexture, sampler uTextureSmplr)
{
    int2 base_coord = (int2(gl_WorkGroupID.xy) * int2(32)) - int2(4);
    uint param = local_index;
    int2 local_coord0 = int2(2) * unswizzle8x8(param);
    int2 coord0 = base_coord + local_coord0;
    int2 param_1 = coord0;
    float2 _178 = generate_mirror_uv(param_1, _104);
    float4 texels0 = uTexture.gather(uTextureSmplr, _178, int2(0), component::x).wzxy;
    int2 param_2 = coord0 + int2(16, 0);
    float2 _188 = generate_mirror_uv(param_2, _104);
    float4 texels1 = uTexture.gather(uTextureSmplr, _188, int2(0), component::x).wzxy;
    int2 param_3 = coord0 + int2(0, 16);
    float2 _197 = generate_mirror_uv(param_3, _104);
    float4 texels2 = uTexture.gather(uTextureSmplr, _197, int2(0), component::x).wzxy;
    int2 param_4 = coord0 + int2(16);
    float2 _206 = generate_mirror_uv(param_4, _104);
    float4 texels3 = uTexture.gather(uTextureSmplr, _206, int2(0), component::x).wzxy;
    if (DCShift)
    {
        texels0 -= float4(0.5);
        texels1 -= float4(0.5);
        texels2 -= float4(0.5);
        texels3 -= float4(0.5);
    }
    int local_coord0_y_half = local_coord0.y >> 1;
    uint param_5 = uint(local_coord0_y_half + 0);
    uint param_6 = uint(local_coord0.x + 0);
    float2 param_7 = texels0.xz;
    store_shared(param_5, param_6, param_7, shared_block);
    uint param_8 = uint(local_coord0_y_half + 0);
    uint param_9 = uint(local_coord0.x + 1);
    float2 param_10 = texels0.yw;
    store_shared(param_8, param_9, param_10, shared_block);
    uint param_11 = uint(local_coord0_y_half + 0);
    uint param_12 = uint(local_coord0.x + 16);
    float2 param_13 = texels1.xz;
    store_shared(param_11, param_12, param_13, shared_block);
    uint param_14 = uint(local_coord0_y_half + 0);
    uint param_15 = uint(local_coord0.x + 17);
    float2 param_16 = texels1.yw;
    store_shared(param_14, param_15, param_16, shared_block);
    uint param_17 = uint(local_coord0_y_half + 8);
    uint param_18 = uint(local_coord0.x + 0);
    float2 param_19 = texels2.xz;
    store_shared(param_17, param_18, param_19, shared_block);
    uint param_20 = uint(local_coord0_y_half + 8);
    uint param_21 = uint(local_coord0.x + 1);
    float2 param_22 = texels2.yw;
    store_shared(param_20, param_21, param_22, shared_block);
    uint param_23 = uint(local_coord0_y_half + 8);
    uint param_24 = uint(local_coord0.x + 16);
    float2 param_25 = texels3.xz;
    store_shared(param_23, param_24, param_25, shared_block);
    uint param_26 = uint(local_coord0_y_half + 8);
    uint param_27 = uint(local_coord0.x + 17);
    float2 param_28 = texels3.yw;
    store_shared(param_26, param_27, param_28, shared_block);
    int2 local_coord = int2(int(32u + (2u * (local_index % 4u))), int(2u * (local_index / 4u)));
    int2 param_29 = base_coord + local_coord;
    float2 _357 = generate_mirror_uv(param_29, _104);
    float4 texels = uTexture.gather(uTextureSmplr, _357, int2(0), component::x).wzxy;
    if (DCShift)
    {
        texels -= float4(0.5);
    }
    uint param_30 = uint(local_coord.y >> 1);
    uint param_31 = uint(local_coord.x + 0);
    float2 param_32 = texels.xz;
    store_shared(param_30, param_31, param_32, shared_block);
    uint param_33 = uint(local_coord.y >> 1);
    uint param_34 = uint(local_coord.x + 1);
    float2 param_35 = texels.yw;
    store_shared(param_33, param_34, param_35, shared_block);
    int2 local_coord_1 = int2(int(2u * (local_index % 16u)), int(32u + (2u * (local_index / 16u))));
    int2 param_36 = base_coord + local_coord_1;
    float2 _411 = generate_mirror_uv(param_36, _104);
    float4 texels_1 = uTexture.gather(uTextureSmplr, _411, int2(0), component::x).wzxy;
    if (DCShift)
    {
        texels_1 -= float4(0.5);
    }
    uint param_37 = uint(local_coord_1.y >> 1);
    uint param_38 = uint(local_coord_1.x + 0);
    float2 param_39 = texels_1.xz;
    store_shared(param_37, param_38, param_39, shared_block);
    uint param_40 = uint(local_coord_1.y >> 1);
    uint param_41 = uint(local_coord_1.x + 1);
    float2 param_42 = texels_1.yw;
    store_shared(param_40, param_41, param_42, shared_block);
    if (local_index < 16u)
    {
        int2 local_coord_2 = int2(int(32u + (2u * (local_index % 4u))), int(32u + (2u * (local_index / 4u))));
        int2 param_43 = base_coord + local_coord_2;
        float2 _469 = generate_mirror_uv(param_43, _104);
        float4 texels_2 = uTexture.gather(uTextureSmplr, _469, int2(0), component::x).wzxy;
        if (DCShift)
        {
            texels_2 -= float4(0.5);
        }
        uint param_44 = uint(local_coord_2.y >> 1);
        uint param_45 = uint(local_coord_2.x + 0);
        float2 param_46 = texels_2.xz;
        store_shared(param_44, param_45, param_46, shared_block);
        uint param_47 = uint(local_coord_2.y >> 1);
        uint param_48 = uint(local_coord_2.x + 1);
        float2 param_49 = texels_2.yw;
        store_shared(param_47, param_48, param_49, shared_block);
    }
}

static inline __attribute__((always_inline))
float2 load_shared(thread const uint& y, thread const uint& x, threadgroup spvUnsafeArray<spvUnsafeArray<float2, 41>, 20>& shared_block)
{
    return shared_block[y][x];
}

static inline __attribute__((always_inline))
void forward_transform8x2(threadgroup spvUnsafeArray<spvUnsafeArray<float2, 41>, 20>& shared_block, thread uint& local_index)
{
    int2 local_coord = int2(int(8u * (local_index % 4u)), int(local_index / 4u));
    spvUnsafeArray<float2, 16> values;
    for (int i = 0; i < 16; i++)
    {
        uint param = uint(local_coord.y);
        uint param_1 = uint(local_coord.x + i);
        float2 v = load_shared(param, param_1, shared_block);
        values[i] = v;
    }
    for (int i_1 = 1; i_1 < 15; i_1 += 2)
    {
        values[i_1] += ((values[i_1 - 1] + values[i_1 + 1]) * (-1.58613431453704833984375));
    }
    for (int i_2 = 2; i_2 < 14; i_2 += 2)
    {
        values[i_2] += ((values[i_2 - 1] + values[i_2 + 1]) * (-0.052980117499828338623046875));
    }
    for (int i_3 = 3; i_3 < 13; i_3 += 2)
    {
        values[i_3] += ((values[i_3 - 1] + values[i_3 + 1]) * 0.88291108608245849609375);
    }
    for (int i_4 = 4; i_4 < 12; i_4 += 2)
    {
        values[i_4] += ((values[i_4 - 1] + values[i_4 + 1]) * 0.4435068666934967041015625);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i_5 = 2; i_5 < 6; i_5++)
    {
        float2 a = values[(2 * i_5) + 0];
        float2 b = values[(2 * i_5) + 1];
        a *= 0.812893092632293701171875;
        b *= 1.23017406463623046875;
        float2 t0 = float2(a.x, b.x);
        float2 t1 = float2(a.y, b.y);
        int y_coord = (local_coord.x >> 1) + (i_5 - 2);
        uint param_2 = uint(y_coord);
        uint param_3 = uint((2 * local_coord.y) + 0);
        float2 param_4 = t0;
        store_shared(param_2, param_3, param_4, shared_block);
        uint param_5 = uint(y_coord);
        uint param_6 = uint((2 * local_coord.y) + 1);
        float2 param_7 = t1;
        store_shared(param_5, param_6, param_7, shared_block);
    }
}

static inline __attribute__((always_inline))
void forward_transform4x2(thread const bool& active_lane, thread const int& y_offset, threadgroup spvUnsafeArray<spvUnsafeArray<float2, 41>, 20>& shared_block, thread uint& local_index)
{
    int2 local_coord = int2(int(4u * (local_index % 8u)), int((local_index / 8u) + uint(y_offset)));
    spvUnsafeArray<float2, 12> values;
    if (active_lane)
    {
        for (int i = 0; i < 12; i++)
        {
            uint param = uint(local_coord.y);
            uint param_1 = uint(local_coord.x + i);
            float2 v = load_shared(param, param_1, shared_block);
            values[i] = v;
        }
        for (int i_1 = 1; i_1 < 11; i_1 += 2)
        {
            values[i_1] += ((values[i_1 - 1] + values[i_1 + 1]) * (-1.58613431453704833984375));
        }
        for (int i_2 = 2; i_2 < 10; i_2 += 2)
        {
            values[i_2] += ((values[i_2 - 1] + values[i_2 + 1]) * (-0.052980117499828338623046875));
        }
        for (int i_3 = 3; i_3 < 9; i_3 += 2)
        {
            values[i_3] += ((values[i_3 - 1] + values[i_3 + 1]) * 0.88291108608245849609375);
        }
        for (int i_4 = 4; i_4 < 8; i_4 += 2)
        {
            values[i_4] += ((values[i_4 - 1] + values[i_4 + 1]) * 0.4435068666934967041015625);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active_lane)
    {
        for (int i_5 = 2; i_5 < 4; i_5++)
        {
            float2 a = values[(2 * i_5) + 0];
            float2 b = values[(2 * i_5) + 1];
            a *= 0.812893092632293701171875;
            b *= 1.23017406463623046875;
            float2 t0 = float2(a.x, b.x);
            float2 t1 = float2(a.y, b.y);
            int y_coord = (local_coord.x >> 1) + (i_5 - 2);
            uint param_2 = uint(y_coord);
            uint param_3 = uint((2 * local_coord.y) + 0);
            float2 param_4 = t0;
            store_shared(param_2, param_3, param_4, shared_block);
            uint param_5 = uint(y_coord);
            uint param_6 = uint((2 * local_coord.y) + 1);
            float2 param_7 = t1;
            store_shared(param_5, param_6, param_7, shared_block);
        }
    }
}

kernel void pyrowave_dwt(constant Registers& _104 [[buffer(0)]], texture2d<float> uTexture [[texture(0)]], texture2d_array<float, access::write> uOutput [[texture(1)]], sampler uTextureSmplr [[sampler(0)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]], uint gl_SubgroupSize [[threads_per_simdgroup]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])
{
    threadgroup spvUnsafeArray<spvUnsafeArray<float2, 41>, 20> shared_block;
    uint local_index = (gl_SubgroupID * gl_SubgroupSize) + gl_SubgroupInvocationID;
    load_image_with_apron(shared_block, _104, gl_WorkGroupID, local_index, uTexture, uTextureSmplr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    forward_transform8x2(shared_block, local_index);
    bool param = local_index < 32u;
    int param_1 = 16;
    forward_transform4x2(param, param_1, shared_block, local_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    forward_transform8x2(shared_block, local_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint param_2 = local_index;
    int2 local_coord = unswizzle8x8(param_2);
    for (int y = local_coord.y; y < 16; y += 8)
    {
        int _978 = local_coord.x * 2;
        for (int x = _978; x < 32; x += 16)
        {
            uint param_3 = uint(y);
            uint param_4 = uint(x + 0);
            float2 v0 = load_shared(param_3, param_4, shared_block);
            uint param_5 = uint(y);
            uint param_6 = uint(x + 1);
            float2 v1 = load_shared(param_5, param_6, shared_block);
            int img_x = x >> 1;
            int img_y = y;
            int2 base_image_coord = (int2(gl_WorkGroupID.xy) * int2(16)) + int2(img_x, img_y);
            int3 _1027 = int3(base_image_coord, 0);
            uOutput.write(v0.xxxx, uint2(_1027.xy), uint(_1027.z));
            int3 _1034 = int3(base_image_coord, 2);
            uOutput.write(v0.yyyy, uint2(_1034.xy), uint(_1034.z));
            int3 _1041 = int3(base_image_coord, 1);
            uOutput.write(v1.xxxx, uint2(_1041.xy), uint(_1041.z));
            int3 _1048 = int3(base_image_coord, 3);
            uOutput.write(v1.yyyy, uint2(_1048.xy), uint(_1048.z));
        }
    }
}

