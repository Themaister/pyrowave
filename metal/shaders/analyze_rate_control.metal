#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

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
inline T spvSubgroupShuffleXor(T value, ushort mask)
{
    return simd_shuffle_xor(value, mask);
}

template<>
inline bool spvSubgroupShuffleXor(bool value, ushort mask)
{
    return !!simd_shuffle_xor((ushort)value, mask);
}

template<uint N>
inline vec<bool, N> spvSubgroupShuffleXor(vec<bool, N> value, ushort mask)
{
    return (vec<bool, N>)simd_shuffle_xor((vec<ushort, N>)value, mask);
}

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

struct RDOperation
{
    int quant;
    uint block_offset_saving;
};

struct Buckets
{
    uint count;
    uint consumed_payload;
    char _m2_pad[56];
    uint total_savings_per_bucket[2048];
    RDOperation rdo_operations[1];
};

struct Registers
{
    int2 resolution;
    int2 resolution_8x8_blocks;
    int block_offset_8x8;
    int block_stride_8x8;
    int block_offset_32x32;
    int block_stride_32x32;
    uint total_wg_count;
    uint num_blocks_aligned;
    uint block_index_shamt;
};

struct RDOperation_1
{
    int quant;
    uint block_offset_saving;
};

struct QuantStats
{
    half square_error;
    ushort payload_cost;
};

struct BlockStats
{
    uint num_planes;
    QuantStats errors[15];
};

struct SSBOBlockStats
{
    BlockStats stats[1];
};

struct QuantStats_1
{
    half square_error;
    ushort payload_cost;
};

static inline __attribute__((always_inline))
uint distortion_to_bucket_index(thread const float& d, thread const float& cost, thread const float& d_base, thread const float& cost_base)
{
    if (cost == cost_base)
    {
        return 0u;
    }
    float index = 60.0 + (2.0 * log2(fast::max(d - d_base, 0.0) / (cost_base - cost)));
    return uint(fast::max(index + 0.5, 0.0));
}

static inline __attribute__((always_inline))
uint inclusive_max_clustered16(thread uint& v, thread uint& gl_SubgroupInvocationID)
{
    v = min(v, (112u + gl_SubgroupInvocationID));
    for (uint i = 1u; i < 16u; i *= 2u)
    {
        uint up = spvSubgroupShuffleUp(v, i) + i;
        v = max(v, ((gl_SubgroupInvocationID >= i) ? up : 0u));
    }
    return v;
}

static inline __attribute__((always_inline))
void emit_rdo_operations(thread uint& gl_SubgroupInvocationID, threadgroup spvUnsafeArray<uint, 16>& shared_rate_cost, threadgroup spvUnsafeArray<float, 16>& shared_distortion, device Buckets& buckets, thread uint3& gl_WorkGroupID, constant Registers& registers)
{
    float cost;
    float distortion;
    if (gl_SubgroupInvocationID < 16u)
    {
        cost = float(shared_rate_cost[gl_SubgroupInvocationID]);
        distortion = shared_distortion[gl_SubgroupInvocationID];
    }
    else
    {
        cost = float(shared_rate_cost[gl_SubgroupInvocationID]);
        distortion = 1000000015047466219876688855040.0;
    }
    float param = distortion;
    float param_1 = cost;
    float param_2 = shared_distortion[0];
    float param_3 = float(shared_rate_cost[0]);
    uint bucket_index = distortion_to_bucket_index(param, param_1, param_2, param_3);
    if (gl_SubgroupInvocationID == 0u)
    {
        bucket_index = 0u;
    }
    uint param_4 = bucket_index;
    uint _139 = inclusive_max_clustered16(param_4, gl_SubgroupInvocationID);
    uint inclusive_bucket_index = _139;
    if (gl_SubgroupInvocationID == 0u)
    {
        uint unquantized_cost = shared_rate_cost[0];
        uint _158 = atomic_fetch_add_explicit((device atomic_uint*)&buckets.consumed_payload, unquantized_cost, memory_order_relaxed);
    }
    else
    {
        if (gl_SubgroupInvocationID < 16u)
        {
            uint saving = shared_rate_cost[gl_SubgroupInvocationID - 1u] - shared_rate_cost[gl_SubgroupInvocationID];
            if (saving != 0u)
            {
                int2 block32x32_index = int2(gl_WorkGroupID.xy);
                int block_index = (registers.block_offset_32x32 + (block32x32_index.y * registers.block_stride_32x32)) + block32x32_index.x;
                uint subdivision = uint(block_index >> int(registers.block_index_shamt));
                uint _221 = atomic_fetch_add_explicit((device atomic_uint*)&buckets.total_savings_per_bucket[(inclusive_bucket_index * 16u) + subdivision], saving, memory_order_relaxed);
                RDOperation_1 _240 = RDOperation_1{ int(gl_SubgroupInvocationID), uint(block_index) | (saving << uint(16)) };
                RDOperation _243;
                _243.quant = _240.quant;
                _243.block_offset_saving = _240.block_offset_saving;
                buckets.rdo_operations[uint(block_index) + (inclusive_bucket_index * registers.num_blocks_aligned)] = _243;
            }
        }
    }
}

kernel void pyrowave_analyze_rate_control(device Buckets& buckets [[buffer(0)]], constant Registers& registers [[buffer(1)]], device SSBOBlockStats& block_stats [[buffer(2)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[threads_per_simdgroup]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]])
{
    threadgroup spvUnsafeArray<uint, 16> shared_rate_cost;
    threadgroup spvUnsafeArray<float, 16> shared_distortion;
    threadgroup spvUnsafeArray<uint, 4> shared_tmp;
    uint index = gl_SubgroupInvocationID + (gl_SubgroupSize * gl_SubgroupID);
    int2 block32x32_index = int2(gl_WorkGroupID.xy);
    int2 local_block_index = int2(int(extract_bits(index, uint(0), uint(2))), int(extract_bits(index, uint(2), uint(2))));
    int2 block8x8_index = (int2(4) * block32x32_index) + local_block_index;
    bool block8x8_in_range = all(block8x8_index < registers.resolution_8x8_blocks);
    int block_index_8x8 = (registers.block_offset_8x8 + (registers.block_stride_8x8 * block8x8_index.y)) + block8x8_index.x;
    uint num_active_planes;
    if (block8x8_in_range)
    {
        num_active_planes = block_stats.stats[block_index_8x8].num_planes;
    }
    uint bit_index = index >> uint(4);
    for (uint i = bit_index; i < 16u; i += 4u)
    {
        float dist = 0.0;
        uint cost = 0u;
        if (block8x8_in_range)
        {
            uint _331 = min(i, num_active_planes);
            QuantStats_1 _335;
            _335.square_error = block_stats.stats[block_index_8x8].errors[_331].square_error;
            _335.payload_cost = block_stats.stats[block_index_8x8].errors[_331].payload_cost;
            QuantStats_1 stats = _335;
            dist = float(stats.square_error);
            cost = uint(stats.payload_cost);
        }
        if (cost != 0u)
        {
            cost += 24u;
        }
        if (gl_SubgroupSize == 16u)
        {
            cost = simd_sum(cost);
            dist = simd_sum(dist);
        }
        else
        {
            cost += spvSubgroupShuffleXor(cost, 1u);
            cost += spvSubgroupShuffleXor(cost, 2u);
            cost += spvSubgroupShuffleXor(cost, 4u);
            cost += spvSubgroupShuffleXor(cost, 8u);
            dist += spvSubgroupShuffleXor(dist, 1u);
            dist += spvSubgroupShuffleXor(dist, 2u);
            dist += spvSubgroupShuffleXor(dist, 4u);
            dist += spvSubgroupShuffleXor(dist, 8u);
        }
        if ((index & 15u) == 0u)
        {
            if (cost != 0u)
            {
                cost += 64u;
            }
            shared_rate_cost[i] = (cost + 31u) >> uint(5);
            shared_distortion[i] = dist;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_SubgroupID == 0u)
    {
        emit_rdo_operations(gl_SubgroupInvocationID, shared_rate_cost, shared_distortion, buckets, gl_WorkGroupID, registers);
    }
}

