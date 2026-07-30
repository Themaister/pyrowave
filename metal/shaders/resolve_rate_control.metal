#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

template<typename T>
inline T spvSubgroupShuffle(T value, ushort lane)
{
    return simd_shuffle(value, lane);
}

template<>
inline bool spvSubgroupShuffle(bool value, ushort lane)
{
    return !!simd_shuffle((ushort)value, lane);
}

template<uint N>
inline vec<bool, N> spvSubgroupShuffle(vec<bool, N> value, ushort lane)
{
    return (vec<bool, N>)simd_shuffle((vec<ushort, N>)value, lane);
}

template<>
inline ulong spvSubgroupShuffle(ulong value, ushort lane)
{
    return as_type<ulong>(spvSubgroupShuffle(as_type<uint2>(value), lane));
}

template<>
inline ulong2 spvSubgroupShuffle(ulong2 value, ushort lane)
{
    return ulong2(spvSubgroupShuffle(value.x, lane), spvSubgroupShuffle(value.y, lane));
}

inline ulong3 spvSubgroupShuffle(ulong3 value, ushort lane)
{
    return ulong3(spvSubgroupShuffle(value.xy, lane), spvSubgroupShuffle(value.z, lane));
}

inline ulong4 spvSubgroupShuffle(ulong4 value, ushort lane)
{
    return ulong4(spvSubgroupShuffle(value.xy, lane), spvSubgroupShuffle(value.zw, lane));
}

template<uint N>
inline vec<long, N> spvSubgroupShuffle(vec<long, N> value, ushort lane)
{
    return vec<long, N>(spvSubgroupShuffle(vec<ulong, N>(value), lane));
}

constant uint _7_tmp [[function_constant(0)]];
constant uint _7 = is_function_constant_defined(_7_tmp) ? _7_tmp : 1u;

struct RDOperation
{
    int quant;
    uint block_offset_saving;
};

struct Buckets
{
    char _m0_pad[4];
    int consumed_payload;
    char _m1_pad[56];
    int total_savings_per_bucket[2048];
    RDOperation rdo_operations[1];
};

struct Registers
{
    uint target_payload_size;
    uint num_blocks_per_subdivision;
};

struct RDOperation_1
{
    int quant;
    uint block_offset_saving;
};

struct QuantList
{
    int data[1];
};

constant uint _169 = is_function_constant_defined(_7_tmp) ? _7_tmp : 1u;
constant uint3 _170 = uint3(_169, 1u, 1u);

kernel void pyrowave_resolve_rate_control(device Buckets& buckets [[buffer(0)]], constant Registers& registers [[buffer(1)]], device QuantList& quant_data [[buffer(2)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]], uint gl_SubgroupSize [[threads_per_simdgroup]])
{
    int required_savings_per_bucket = buckets.consumed_payload - int(registers.target_payload_size);
    if (gl_WorkGroupID.x != 0u)
    {
        int prev_bucket_total = buckets.total_savings_per_bucket[gl_WorkGroupID.x - 1u];
        if (buckets.total_savings_per_bucket[gl_WorkGroupID.x] == prev_bucket_total)
        {
            return;
        }
        required_savings_per_bucket -= prev_bucket_total;
    }
    else
    {
        if (buckets.total_savings_per_bucket[gl_WorkGroupID.x] == 0)
        {
            return;
        }
    }
    if (required_savings_per_bucket <= 0)
    {
        return;
    }
    uint total_saved = 0u;
    uint i = 0u;
    for (;;)
    {
        bool _86 = i < registers.num_blocks_per_subdivision;
        bool _93;
        if (_86)
        {
            _93 = total_saved < uint(required_savings_per_bucket);
        }
        else
        {
            _93 = _86;
        }
        if (_93)
        {
            RDOperation_1 op = RDOperation_1{ 0, 0u };
            if ((i + gl_SubgroupInvocationID) < registers.num_blocks_per_subdivision)
            {
                uint _116 = ((gl_WorkGroupID.x * registers.num_blocks_per_subdivision) + i) + gl_SubgroupInvocationID;
                RDOperation_1 _120;
                _120.quant = buckets.rdo_operations[_116].quant;
                _120.block_offset_saving = buckets.rdo_operations[_116].block_offset_saving;
                op = _120;
            }
            uint saving = extract_bits(op.block_offset_saving, uint(16), uint(16));
            uint block_offset = extract_bits(op.block_offset_saving, uint(0), uint(16));
            uint scan_saving = simd_prefix_inclusive_sum(saving);
            bool should_apply_quant = ((total_saved + scan_saving) - saving) < uint(required_savings_per_bucket);
            if (should_apply_quant && (saving != 0u))
            {
                int _158 = atomic_fetch_max_explicit((device atomic_int*)&quant_data.data[block_offset], op.quant, memory_order_relaxed);
            }
            total_saved += spvSubgroupShuffle(scan_saving, gl_SubgroupSize - 1u);
            i += gl_SubgroupSize;
            continue;
        }
        else
        {
            break;
        }
    }
}

