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

template<typename T>
inline T spvSubgroupShuffleUp(T value, ushort delta)
{
    return simd_shuffle_up(value, delta);
}

template<>
inline bool spvSubgroupShuffleUp(bool value, ushort delta)
{
    return !!simd_shuffle_up((ushort)value, delta);
}

template<uint N>
inline vec<bool, N> spvSubgroupShuffleUp(vec<bool, N> value, ushort delta)
{
    return (vec<bool, N>)simd_shuffle_up((vec<ushort, N>)value, delta);
}

struct Payloads8
{
    uchar data[1];
};

struct Registers
{
    int2 resolution;
    int output_layer;
    int block_offset_32x32;
    int block_stride_32x32;
};

struct PayloadOffsets
{
    uint data[1];
};

struct Payloads
{
    uint data[1];
};

struct Payloads16
{
    ushort data[1];
};

constant spvUnsafeArray<int, 8> _143 = spvUnsafeArray<int, 8>({ 0, 0, 0, 0, 0, 0, 0, 0 });

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
float2x4 decode_payload(thread const uint& code_word, thread const uint& q_bits, thread const uint& offset, thread const uint& block_index, const device Payloads8& payload_data_u8)
{
    bool empty_block = code_word == 0u;
    if (empty_block)
    {
        return float2x4(float4(0.0), float4(0.0));
    }
    int bit_offset = 2 * int(block_index);
    uint lsbs = code_word & 21845u;
    uint msbs = code_word & 43690u;
    uint msbs_shift = msbs >> uint(1);
    msbs |= msbs_shift;
    uint byte_offset = (uint(int(popcount(extract_bits(lsbs, uint(0), uint(bit_offset)))) + int(popcount(extract_bits(msbs, uint(0), uint(bit_offset))))) + (q_bits * block_index)) + offset;
    uint payload = uint(payload_data_u8.data[byte_offset]);
    uint local_control_word = extract_bits(code_word, uint(bit_offset), uint(2));
    spvUnsafeArray<int, 8> decoded_abs = spvUnsafeArray<int, 8>({ 0, 0, 0, 0, 0, 0, 0, 0 });
    int plane_iterations = int(q_bits + local_control_word);
    int _151 = plane_iterations - 1;
    for (int q = _151; q >= 0; q--)
    {
        for (int b = 0; b < 8; b++)
        {
            int decoded = int(extract_bits(payload, uint(b), uint(1)));
            decoded_abs[b] = insert_bits(decoded_abs[b], decoded, uint(q), uint(1));
        }
        byte_offset++;
        payload = uint(payload_data_u8.data[byte_offset]);
    }
    float2x4 m;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            float v = float(decoded_abs[(i * 2) + j]);
            if (v != 0.0)
            {
                v += 0.5;
            }
            m[j][i] = v;
        }
    }
    return m;
}

static inline __attribute__((always_inline))
float decode_quant(thread const uint& quant_code)
{
    int e = 4 - int(quant_code >> uint(3));
    int m = int(quant_code) & 7;
    float inv_quant = 1.1920928955078125e-07 * float((8 + m) * (1 << (20 + e)));
    return inv_quant;
}

static inline __attribute__((always_inline))
float decode_quant_scale(thread const uint& code)
{
    return (float(code) / 8.0) + 0.25;
}

static inline __attribute__((always_inline))
uint scan_subgroups(thread uint& v, thread uint& gl_NumSubgroups, thread uint& gl_SubgroupInvocationID)
{
    for (uint i = 1u; i < gl_NumSubgroups; i *= 2u)
    {
        uint up = spvSubgroupShuffleUp(v, i);
        v += ((gl_SubgroupInvocationID >= i) ? up : 0u);
    }
    return v;
}

static inline __attribute__((always_inline))
void scan_subgroups_fallback(thread const uint& local_index, thread uint& gl_NumSubgroups, threadgroup spvUnsafeArray<uint, 32>& shared_sign_scan)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool active_lane = local_index < gl_NumSubgroups;
    uint v = 0u;
    if (active_lane)
    {
        v = shared_sign_scan[local_index];
    }
    for (uint i = 1u; i < gl_NumSubgroups; i *= 2u)
    {
        uint up = 0u;
        bool do_work = (local_index >= i) && active_lane;
        if (do_work)
        {
            up = shared_sign_scan[local_index - i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (do_work)
        {
            v += up;
            shared_sign_scan[local_index] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void pyrowave_wavelet_dequant(const device void* spvBufferAliasSet0Binding2 [[buffer(0)]], constant Registers& registers [[buffer(1)]], const device PayloadOffsets& payload_offsets [[buffer(2)]], texture2d_array<float, access::write> uDequantImg [[texture(0)]], uint gl_NumSubgroups [[simdgroups_per_threadgroup]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]], uint gl_SubgroupSize [[threads_per_simdgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    const device auto& payload_data_u8 = *(const device Payloads8*)spvBufferAliasSet0Binding2;
    const device auto& payload_data_u32 = *(const device Payloads*)spvBufferAliasSet0Binding2;
    const device auto& payload_data_u16 = *(const device Payloads16*)spvBufferAliasSet0Binding2;
    threadgroup spvUnsafeArray<uint, 32> shared_sign_scan;
    threadgroup uint shared_sign_offset;
    threadgroup spvUnsafeArray<uint, 16> shared_plane_byte_offsets;
    uint local_index = (gl_SubgroupID * gl_SubgroupSize) + gl_SubgroupInvocationID;
    int block_index_32x32 = int((uint(registers.block_offset_32x32) + (gl_WorkGroupID.y * uint(registers.block_stride_32x32))) + gl_WorkGroupID.x);
    uint block_local_index = extract_bits(local_index, uint(0), uint(3));
    uint block_x = extract_bits(local_index, uint(3), uint(2));
    uint block_y = extract_bits(local_index, uint(5), uint(2));
    uint linear_block = (block_y * 4u) + block_x;
    uint param = block_local_index << uint(3);
    int2 local_coord = unswizzle8x8(param);
    int2 coord = int2(gl_WorkGroupID.xy) * int2(32);
    coord += (int2(8) * int2(int(block_x), int(block_y)));
    coord += local_coord;
    uint offset_u32 = payload_offsets.data[block_index_32x32];
    if (offset_u32 == 4294967295u)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                int3 _459 = int3(coord + int2(i, j), registers.output_layer);
                uDequantImg.write(float4(0.0), uint2(_459.xy), uint(_459.z));
            }
        }
        return;
    }
    uint ballot = payload_data_u32.data[offset_u32] & 65535u;
    uint q_code = payload_data_u32.data[offset_u32 + 1u] & 255u;
    if (local_index < 16u)
    {
        uint control_word = 0u;
        uint q_bits = 0u;
        if (extract_bits(ballot, uint(int(local_index)), uint(1)) != 0u)
        {
            uint local_code_offset = uint(int(popcount(extract_bits(ballot, uint(0), uint(int(local_index))))));
            control_word = uint(payload_data_u16.data[((offset_u32 * 2u) + 4u) + local_code_offset]);
            q_bits = uint(payload_data_u8.data[(((offset_u32 * 4u) + 8u) + uint(int(popcount(ballot)) * 2)) + local_code_offset]) & 15u;
        }
        uint lsbs = control_word & 21845u;
        uint msbs = control_word & 43690u;
        uint msbs_shift = msbs >> uint(1);
        msbs |= msbs_shift;
        uint byte_cost = uint(int(popcount(lsbs)) + int(popcount(msbs))) + (q_bits * 8u);
        uint byte_scan = (((offset_u32 * 4u) + 8u) + uint(3 * int(popcount(ballot)))) + simd_prefix_inclusive_sum(byte_cost);
        if (local_index == 15u)
        {
            shared_sign_offset = 8u * byte_scan;
        }
        shared_plane_byte_offsets[local_index] = byte_scan - byte_cost;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2x4 v;
    int significant_count;
    if (extract_bits(ballot, uint(int(linear_block)), uint(1)) != 0u)
    {
        uint local_code_offset_1 = uint(int(popcount(extract_bits(ballot, uint(0), uint(int(linear_block))))));
        uint control_word_1 = uint(payload_data_u16.data[((offset_u32 * 2u) + 4u) + local_code_offset_1]);
        uint control_word2 = uint(payload_data_u8.data[(((offset_u32 * 4u) + 8u) + uint(int(popcount(ballot)) * 2)) + local_code_offset_1]);
        uint param_1 = control_word_1;
        uint param_2 = control_word2 & 15u;
        uint param_3 = shared_plane_byte_offsets[linear_block];
        uint param_4 = block_local_index;
        v = decode_payload(param_1, param_2, param_3, param_4, payload_data_u8);
        significant_count = 0;
        for (int j_1 = 0; j_1 < 2; j_1++)
        {
            for (int i_1 = 0; i_1 < 4; i_1++)
            {
                significant_count += int(v[j_1][i_1] != 0.0);
            }
        }
        uint param_5 = q_code;
        float q = decode_quant(param_5);
        uint param_6 = extract_bits(control_word2, uint(4), uint(4));
        float inv_scale = q * decode_quant_scale(param_6);
        v = v * inv_scale;
    }
    else
    {
        v = float2x4(float4(0.0), float4(0.0));
        significant_count = 0;
    }
    int significant_scan = simd_prefix_inclusive_sum(significant_count);
    if (gl_SubgroupInvocationID == (gl_SubgroupSize - 1u))
    {
        shared_sign_scan[gl_SubgroupID] = uint(significant_scan);
    }
    if (gl_NumSubgroups <= 8u)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gl_SubgroupSize <= 32u)
        {
            if (local_index < gl_NumSubgroups)
            {
                shared_sign_scan[local_index] = simd_prefix_inclusive_sum(shared_sign_scan[local_index]);
            }
        }
        else
        {
            if (local_index < gl_NumSubgroups)
            {
                uint param_7 = shared_sign_scan[local_index];
                uint _718 = scan_subgroups(param_7, gl_NumSubgroups, gl_SubgroupInvocationID);
                shared_sign_scan[local_index] = _718;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    else
    {
        uint param_8 = local_index;
        scan_subgroups_fallback(param_8, gl_NumSubgroups, shared_sign_scan);
    }
    uint sign_offset = (shared_sign_offset + uint(significant_scan)) - uint(significant_count);
    if (gl_SubgroupID != 0u)
    {
        sign_offset += shared_sign_scan[gl_SubgroupID - 1u];
    }
    uint sign_word = payload_data_u32.data[(sign_offset / 32u) + 0u];
    uint sign_word_upper = payload_data_u32.data[(sign_offset / 32u) + 1u];
    uint masked_sign_offset = sign_offset & 31u;
    if (masked_sign_offset != 0u)
    {
        sign_word = sign_word >> masked_sign_offset;
        sign_word |= (sign_word_upper << (32u - masked_sign_offset));
    }
    int sign_counter = 0;
    for (int i_2 = 0; i_2 < 4; i_2++)
    {
        for (int j_2 = 0; j_2 < 2; j_2++)
        {
            if (v[j_2][i_2] != 0.0)
            {
                v[j_2][i_2] *= (1.0 - (2.0 * float(extract_bits(sign_word, uint(sign_counter), uint(1)))));
                sign_counter++;
            }
        }
    }
    for (int j_3 = 0; j_3 < 2; j_3++)
    {
        for (int i_3 = 0; i_3 < 4; i_3++)
        {
            int3 _841 = int3(coord + int2(i_3, j_3), registers.output_layer);
            uDequantImg.write(float4(v[j_3][i_3]), uint2(_841.xy), uint(_841.z));
        }
    }
}

