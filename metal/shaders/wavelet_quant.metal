#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

// Implementation of the signed GLSL findMSB() function
template<typename T>
inline T spvFindSMSB(T x)
{
    T v = select(x, T(-1) - x, x < T(0));
    return select(clz(T(0)) - (clz(v) + T(1)), T(-1), v == T(0));
}

template<typename T>
[[clang::optnone]] T spvFMul(T l, T r)
{
    return fma(l, r, T(0));
}

template<typename T, int Cols, int Rows>
[[clang::optnone]] vec<T, Cols> spvFMulVectorMatrix(vec<T, Rows> v, matrix<T, Cols, Rows> m)
{
    vec<T, Cols> res = vec<T, Cols>(0);
    for (uint i = Rows; i > 0; --i)
    {
        vec<T, Cols> tmp(0);
        for (uint j = 0; j < Cols; ++j)
        {
            tmp[j] = m[j][i - 1];
        }
        res = fma(tmp, vec<T, Cols>(v[i - 1]), res);
    }
    return res;
}

template<typename T, int Cols, int Rows>
[[clang::optnone]] vec<T, Rows> spvFMulMatrixVector(matrix<T, Cols, Rows> m, vec<T, Cols> v)
{
    vec<T, Rows> res = vec<T, Rows>(0);
    for (uint i = Cols; i > 0; --i)
    {
        res = fma(m[i - 1], vec<T, Rows>(v[i - 1]), res);
    }
    return res;
}

template<typename T, int LCols, int LRows, int RCols, int RRows>
[[clang::optnone]] matrix<T, RCols, LRows> spvFMulMatrixMatrix(matrix<T, LCols, LRows> l, matrix<T, RCols, RRows> r)
{
    matrix<T, RCols, LRows> res;
    for (uint i = 0; i < RCols; i++)
    {
        vec<T, RCols> tmp(0);
        for (uint j = 0; j < LCols; j++)
        {
            tmp = fma(vec<T, RCols>(r[i][j]), l[j], tmp);
        }
        res[i] = tmp;
    }
    return res;
}

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

template<uint N, uint offset>
struct spvClusteredAddDetail;

// Base cases
template<>
struct spvClusteredAddDetail<1, 0>
{
    template<typename T>
    static T op(T value, uint)
    {
        return value;
    }
};

template<uint offset>
struct spvClusteredAddDetail<1, offset>
{
    template<typename T>
    static T op(T value, uint lid)
    {
        // If the target lane is inactive, then return identity.
        if (!extract_bits(as_type<uint2>((simd_vote::vote_t)simd_active_threads_mask())[(lid ^ offset) / 32], (lid ^ offset) % 32, 1))
            return 0;
        return simd_shuffle_xor(value, offset);
    }
};

template<>
struct spvClusteredAddDetail<4, 0>
{
    template<typename T>
    static T op(T value, uint)
    {
        return quad_sum(value);
    }
};

template<uint offset>
struct spvClusteredAddDetail<4, offset>
{
    template<typename T>
    static T op(T value, uint lid)
    {
        // Here, we care if any of the lanes in the quad are active.
        uint quad_mask = extract_bits(as_type<uint2>((simd_vote::vote_t)simd_active_threads_mask())[(lid ^ offset) / 32], ((lid ^ offset) % 32) & ~3, 4);
        if (!quad_mask)
            return 0;
        // But we need to make sure we shuffle from an active lane.
        return simd_shuffle(quad_sum(value), ((lid ^ offset) & ~3) | ctz(quad_mask));
    }
};

// General case
template<uint N, uint offset>
struct spvClusteredAddDetail
{
    template<typename T>
    static T op(T value, uint lid)
    {
        return spvClusteredAddDetail<N/2, offset>::op(value, lid) + spvClusteredAddDetail<N/2, offset + N/2>::op(value, lid);
    }
};

template<uint N, typename T>
T spvClustered_sum(T value, uint lid)
{
    return spvClusteredAddDetail<N, 0>::op(value, lid);
}

template<uint N, uint offset>
struct spvClusteredMaxDetail;

// Base cases
template<>
struct spvClusteredMaxDetail<1, 0>
{
    template<typename T>
    static T op(T value, uint)
    {
        return value;
    }
};

template<uint offset>
struct spvClusteredMaxDetail<1, offset>
{
    template<typename T>
    static T op(T value, uint lid)
    {
        // If the target lane is inactive, then return identity.
        if (!extract_bits(as_type<uint2>((simd_vote::vote_t)simd_active_threads_mask())[(lid ^ offset) / 32], (lid ^ offset) % 32, 1))
            return numeric_limits<T>::min();
        return simd_shuffle_xor(value, offset);
    }
};

template<>
struct spvClusteredMaxDetail<4, 0>
{
    template<typename T>
    static T op(T value, uint)
    {
        return quad_max(value);
    }
};

template<uint offset>
struct spvClusteredMaxDetail<4, offset>
{
    template<typename T>
    static T op(T value, uint lid)
    {
        // Here, we care if any of the lanes in the quad are active.
        uint quad_mask = extract_bits(as_type<uint2>((simd_vote::vote_t)simd_active_threads_mask())[(lid ^ offset) / 32], ((lid ^ offset) % 32) & ~3, 4);
        if (!quad_mask)
            return numeric_limits<T>::min();
        // But we need to make sure we shuffle from an active lane.
        return simd_shuffle(quad_max(value), ((lid ^ offset) & ~3) | ctz(quad_mask));
    }
};

// General case
template<uint N, uint offset>
struct spvClusteredMaxDetail
{
    template<typename T>
    static T op(T value, uint lid)
    {
        return max(spvClusteredMaxDetail<N/2, offset>::op(value, lid), spvClusteredMaxDetail<N/2, offset + N/2>::op(value, lid));
    }
};

template<uint N, typename T>
T spvClustered_max(T value, uint lid)
{
    return spvClusteredMaxDetail<N, 0>::op(value, lid);
}

struct QuantResult
{
    float square_error;
    int encode_cost_early;
    int block4x2_shifted;
    int encode_cost_late_bits;
    int quality_planes;
};

constant bool SkipQuantScale_tmp [[function_constant(1)]];
constant bool SkipQuantScale = is_function_constant_defined(SkipQuantScale_tmp) ? SkipQuantScale_tmp : false;

struct ResType
{
    float _m0;
    int _m1;
};

struct Registers
{
    int2 resolution;
    int2 resolution_8x8_blocks;
    float2 inv_resolution;
    float input_layer;
    float quant_resolution;
    int block_offset;
    int block_stride;
    float rdo_distortion_scale;
};

struct BlockMeta
{
    uint code_word;
    uint offset;
};

struct SSBOMeta
{
    BlockMeta meta[1];
};

struct BlockMeta_1
{
    uint code_word;
    uint offset;
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

struct Payloads
{
    uint counter;
    char _m1_pad[4];
    uchar data[1];
};

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
float max4(thread const float4& v)
{
    float2 v2 = fast::max(v.xy, v.zw);
    return fast::max(v2.x, v2.y);
}

static inline __attribute__((always_inline))
uint encode_quant_scale(thread const float& scale)
{
    return uint(ceil((scale - 0.25) * 8.0));
}

static inline __attribute__((always_inline))
float decode_quant_scale(thread const uint& code)
{
    return (float(code) / 8.0) + 0.25;
}

static inline __attribute__((always_inline))
void compute_quant_scale(thread const float& max_wave_texels, thread uint& quant_code, thread float& quant_scale)
{
    if (SkipQuantScale || (max_wave_texels < 1.0))
    {
        quant_code = 6u;
        quant_scale = 1.0;
    }
    else
    {
        ResType _169;
        _169._m0 = frexp(max_wave_texels - 0.25, _169._m1);
        int e = _169._m1;
        float target_max = float(1 << e) - 0.25;
        float inv_scale = max_wave_texels / target_max;
        float param = inv_scale;
        quant_code = encode_quant_scale(param);
        uint param_1 = quant_code;
        quant_scale = 1.0 / decode_quant_scale(param_1);
    }
}

static inline __attribute__((always_inline))
float compute_square_error(thread float2x4& v, thread const int& q, thread uint& num_significant_values, thread uint& gl_SubgroupInvocationID, constant Registers& registers)
{
    v = float2x4(float4(abs(v[0])), float4(abs(v[1])));
    float2x4 iv = float2x4(float4(floor(ldexp(v[0], int4(-q)))), float4(trunc(ldexp(v[1], int4(-q)))));
    num_significant_values = 0u;
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 4; i++)
        {
            if (iv[j][i] != 0.0)
            {
                num_significant_values++;
            }
        }
    }
    iv[0] += select(float4(0.0), float4(0.5), iv[0] != float4(0.0));
    iv[1] += select(float4(0.0), float4(0.5), iv[1] != float4(0.0));
    iv = float2x4(float4(trunc(ldexp(iv[0], int4(q)))), float4(trunc(ldexp(iv[1], int4(q)))));
    float2x4 err = float2x4(v[0] - iv[0], v[1] - iv[1]);
    num_significant_values = spvClustered_sum<8>(num_significant_values, gl_SubgroupInvocationID);
    return (dot(err[0], err[0]) + dot(err[1], err[1])) * registers.rdo_distortion_scale;
}

static inline __attribute__((always_inline))
QuantResult compute_quant_stats(thread const float2x4& v, thread const int& q, thread int& msb, thread int& block4x2_max, thread const float& inv_quant_squared, thread uint& gl_SubgroupInvocationID, constant Registers& registers)
{
    block4x2_max = block4x2_max >> q;
    float2x4 param = v;
    int param_1 = q;
    uint param_2;
    float _350 = compute_square_error(param, param_1, param_2, gl_SubgroupInvocationID, registers);
    uint wave8_num_significants = param_2;
    QuantResult result;
    result.square_error = _350 * inv_quant_squared;
    result.block4x2_shifted = block4x2_max;
    result.encode_cost_early = int(block4x2_max > 0);
    msb -= q;
    result.quality_planes = 0;
    if (msb >= 3)
    {
        result.quality_planes = msb - 2;
        result.encode_cost_early = result.quality_planes + 1;
        result.block4x2_shifted = result.block4x2_shifted >> result.quality_planes;
    }
    result.encode_cost_early += (spvFindSMSB(result.block4x2_shifted) + 1);
    result.encode_cost_late_bits = (8 * spvClustered_sum<8>(max((result.encode_cost_early - 1), 0), gl_SubgroupInvocationID)) + int(wave8_num_significants);
    return result;
}

static inline __attribute__((always_inline))
int scan_clustered8(thread int& v, thread uint& gl_SubgroupInvocationID)
{
    for (uint i = 1u; i < 8u; i *= 2u)
    {
        int up = spvSubgroupShuffleUp(v, i);
        v += (((gl_SubgroupInvocationID & 7u) >= i) ? up : 0);
    }
    return v;
}

static inline __attribute__((always_inline))
void encode_payload(thread const int2& block_index_8x8, thread float2x4& texels, thread uint& gl_SubgroupInvocationID, constant Registers& registers, device SSBOMeta& block_meta, device SSBOBlockStats& block_stats, device Payloads& payload_data)
{
    float4 param = abs(texels[0]);
    float4 param_1 = abs(texels[1]);
    float max_subblock_texel = fast::max(max4(param), max4(param_1));
    float max_wave_texels = spvClustered_max<8>(max_subblock_texel, gl_SubgroupInvocationID);
    float param_2 = max_wave_texels;
    uint param_3;
    float param_4;
    compute_quant_scale(param_2, param_3, param_4);
    uint quant_code = param_3;
    float quant_scale = param_4;
    texels = texels * quant_scale;
    max_wave_texels = spvFMul(max_wave_texels, quant_scale);
    max_subblock_texel = spvFMul(max_subblock_texel, quant_scale);
    float overall_quant_scale = registers.quant_resolution * quant_scale;
    float inv_quant = 1.0 / overall_quant_scale;
    float inv_quant_squared = inv_quant * inv_quant;
    int4 abs_quant_texels0 = abs(int4(texels[0]));
    int4 abs_quant_texels1 = abs(int4(texels[1]));
    int max_absolute_value = int(max_wave_texels);
    int block4x2_max = int(max_subblock_texel);
    uint block_index = uint((registers.block_offset + (block_index_8x8.y * registers.block_stride)) + block_index_8x8.x);
    if (max_absolute_value == 0)
    {
        if ((gl_SubgroupInvocationID & 7u) == 0u)
        {
            BlockMeta _500;
            _500.code_word = (BlockMeta_1{ 0u, 0u }).code_word;
            _500.offset = (BlockMeta_1{ 0u, 0u }).offset;
            block_meta.meta[block_index] = _500;
            block_stats.stats[block_index].num_planes = 0u;
            QuantStats _521;
            _521.square_error = (QuantStats_1{ half(0.0), ushort(0) }).square_error;
            _521.payload_cost = (QuantStats_1{ half(0.0), ushort(0) }).payload_cost;
            block_stats.stats[block_index].errors[0] = _521;
        }
        return;
    }
    int msb = spvFindSMSB(max_absolute_value);
    float2x4 param_5 = texels;
    int param_6 = 0;
    int param_7 = msb;
    int param_8 = block4x2_max;
    float param_9 = inv_quant_squared;
    QuantResult _536 = compute_quant_stats(param_5, param_6, param_7, param_8, param_9, gl_SubgroupInvocationID, registers);
    QuantResult result = _536;
    int param_10 = result.encode_cost_early;
    int _541 = scan_clustered8(param_10, gl_SubgroupInvocationID);
    int scan = _541;
    uint global_offset = 0u;
    if ((gl_SubgroupInvocationID & 7u) == 7u)
    {
        uint _556 = atomic_fetch_add_explicit((device atomic_uint*)&payload_data.counter, uint(scan), memory_order_relaxed);
        global_offset = _556;
    }
    global_offset = spvSubgroupShuffle(global_offset, gl_SubgroupInvocationID | 7u);
    scan -= result.encode_cost_early;
    int quality_planes = result.quality_planes;
    uint code_word = uint(quality_planes << 16);
    code_word = insert_bits(code_word, quant_code, uint(20), uint(4));
    uint plane_code = uint(spvFindSMSB(result.block4x2_shifted) + 1);
    uint merged_plane_code = plane_code << ((gl_SubgroupInvocationID & 7u) * 2u);
    merged_plane_code |= spvSubgroupShuffleXor(merged_plane_code, 1u);
    merged_plane_code |= spvSubgroupShuffleXor(merged_plane_code, 2u);
    merged_plane_code |= spvSubgroupShuffleXor(merged_plane_code, 4u);
    code_word |= merged_plane_code;
    if ((gl_SubgroupInvocationID & 7u) == 0u)
    {
        BlockMeta_1 _613 = BlockMeta_1{ code_word, global_offset };
        BlockMeta _615;
        _615.code_word = _613.code_word;
        _615.offset = _613.offset;
        block_meta.meta[block_index] = _615;
        block_stats.stats[block_index].num_planes = uint(msb + 1);
        QuantStats_1 _627 = QuantStats_1{ half(0.0), ushort(short(result.encode_cost_late_bits)) };
        QuantStats _629;
        _629.square_error = _627.square_error;
        _629.payload_cost = _627.payload_cost;
        block_stats.stats[block_index].errors[0] = _629;
    }
    for (int q = 1; q <= msb; q++)
    {
        float2x4 param_11 = texels;
        int param_12 = q;
        int param_13 = msb;
        int param_14 = block4x2_max;
        float param_15 = inv_quant_squared;
        QuantResult _650 = compute_quant_stats(param_11, param_12, param_13, param_14, param_15, gl_SubgroupInvocationID, registers);
        QuantResult quant_result = _650;
        float square_error = spvClustered_sum<8>(quant_result.square_error, gl_SubgroupInvocationID);
        if ((gl_SubgroupInvocationID & 7u) == 0u)
        {
            QuantStats_1 _670 = QuantStats_1{ half(fast::min(square_error, 60000.0)), ushort(short(quant_result.encode_cost_late_bits)) };
            QuantStats _672;
            _672.square_error = _670.square_error;
            _672.payload_cost = _670.payload_cost;
            block_stats.stats[block_index].errors[q] = _672;
        }
    }
    float square_error_1 = spvClustered_sum<8>((dot(texels[0], texels[0]) + dot(texels[1], texels[1])) * inv_quant_squared, gl_SubgroupInvocationID);
    if ((gl_SubgroupInvocationID & 7u) == 0u)
    {
        QuantStats_1 _701 = QuantStats_1{ half(fast::min(60000.0, square_error_1)), ushort(0) };
        QuantStats _703;
        _703.square_error = _701.square_error;
        _703.payload_cost = _701.payload_cost;
        block_stats.stats[block_index].errors[msb + 1] = _703;
    }
    uint byte_offset = uint(scan) + global_offset;
    bool need_sign = (result.block4x2_shifted != 0) || (quality_planes != 0);
    if (need_sign)
    {
        uint4 s0 = uint4(texels[0] < float4(0.0)) << uint4(0u, 1u, 2u, 3u);
        uint4 s1 = uint4(texels[1] < float4(0.0)) << uint4(4u, 5u, 6u, 7u);
        uint s = ((((((s0.x | s0.y) | s0.z) | s0.w) | s1.x) | s1.y) | s1.z) | s1.w;
        uint _763 = byte_offset;
        byte_offset = _763 + uint(1);
        payload_data.data[_763] = uchar(s);
        int plane_iterations = quality_planes + int(plane_code);
        int _776 = plane_iterations - 1;
        int q_1 = _776;
        do
        {
            s0 = uint4(extract_bits(uint(abs_quant_texels0.x), uint(q_1), uint(1)), extract_bits(uint(abs_quant_texels0.y), uint(q_1), uint(1)), extract_bits(uint(abs_quant_texels0.z), uint(q_1), uint(1)), extract_bits(uint(abs_quant_texels0.w), uint(q_1), uint(1)));
            s1 = uint4(extract_bits(uint(abs_quant_texels1.x), uint(q_1), uint(1)), extract_bits(uint(abs_quant_texels1.y), uint(q_1), uint(1)), extract_bits(uint(abs_quant_texels1.z), uint(q_1), uint(1)), extract_bits(uint(abs_quant_texels1.w), uint(q_1), uint(1)));
            s0 = s0 << uint4(0u, 1u, 2u, 3u);
            s1 = s1 << uint4(4u, 5u, 6u, 7u);
            s = ((((((s0.x | s0.y) | s0.z) | s0.w) | s1.x) | s1.y) | s1.z) | s1.w;
            uint _850 = byte_offset;
            byte_offset = _850 + uint(1);
            payload_data.data[_850] = uchar(s);
            q_1--;
        } while (q_1 >= 0);
    }
}

kernel void pyrowave_wavelet_quant(constant Registers& registers [[buffer(0)]], device SSBOMeta& block_meta [[buffer(1)]], device SSBOBlockStats& block_stats [[buffer(2)]], device Payloads& payload_data [[buffer(3)]], texture2d_array<float> uTexture [[texture(0)]], sampler uTextureSmplr [[sampler(0)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]], uint gl_SubgroupSize [[threads_per_simdgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    uint local_index = (gl_SubgroupID * gl_SubgroupSize) + gl_SubgroupInvocationID;
    uint block_local_index = extract_bits(local_index, uint(0), uint(3));
    uint block_x = extract_bits(local_index, uint(3), uint(2));
    uint block_y = extract_bits(local_index, uint(5), uint(2));
    uint param = block_local_index << uint(3);
    int2 local_coord = unswizzle8x8(param);
    int2 coord = int2(gl_WorkGroupID.xy) * int2(32);
    coord += (int2(8) * int2(int(block_x), int(block_y)));
    coord += local_coord;
    int2 block_index = (int2(4) * int2(gl_WorkGroupID.xy)) + int2(int(block_x), int(block_y));
    float3 uv = float3(float2(coord) * registers.inv_resolution, registers.input_layer);
    float4 texels0 = uTexture.gather(uTextureSmplr, uv.xy, uint(rint(uv.z)), int2(1), component::x).wxzy;
    float4 texels1 = uTexture.gather(uTextureSmplr, uv.xy, uint(rint(uv.z)), int2(3, 1), component::x).wxzy;
    float4 scaled_texels0 = texels0 * registers.quant_resolution;
    float4 scaled_texels1 = texels1 * registers.quant_resolution;
    bool in_bounds = all(block_index < registers.resolution_8x8_blocks);
    if (in_bounds)
    {
        int2 param_1 = block_index;
        float2x4 param_2 = float2x4(float4(scaled_texels0), float4(scaled_texels1));
        encode_payload(param_1, param_2, gl_SubgroupInvocationID, registers, block_meta, block_stats, payload_data);
    }
}

