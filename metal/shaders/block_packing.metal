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

// Implementation of the GLSL findLSB() function
template<typename T>
inline T spvFindLSB(T x)
{
    return select(ctz(x), T(-1), x == T(0));
}

inline uint4 spvSubgroupBallot(bool value)
{
    simd_vote vote = simd_ballot(value);
    // simd_ballot() returns a 64-bit integer-like object, but
    // SPIR-V callers expect a uint4. We must convert.
    // FIXME: This won't include higher bits if Apple ever supports
    // 128 lanes in an SIMD-group.
    return uint4(as_type<uint2>((simd_vote::vote_t)vote), 0, 0);
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

struct Payloads
{
    char _m0_pad[4];
    uint bitstream_payload_counter;
    uchar data[1];
};

struct BitstreamPayload8Bit
{
    uchar data[1];
};

struct Registers
{
    int2 resolution;
    int2 resolution_32x32_blocks;
    int2 resolution_8x8_blocks;
    uint quant_resolution_code;
    uint sequence_code;
    int block_offset_32x32;
    int block_stride_32x32;
    int block_offset_8x8;
    int block_stride_8x8;
};

struct RateControlQuant
{
    int data[1];
};

struct BlockMeta
{
    uint code_word;
    uint offset;
};

struct BlockMeta_1
{
    uint code_word;
    uint offset;
};

struct SSBOMeta
{
    BlockMeta_1 meta[1];
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

struct BitstreamPayload
{
    uint data[1];
};

struct BitstreamPacket
{
    uint offset;
    uint num_words;
};

struct BitstreamMeta
{
    BitstreamPacket packets[1];
};

struct BitstreamPacket_1
{
    uint offset;
    uint num_words;
};

struct BitstreamPayload16Bit
{
    ushort data[1];
};

static inline __attribute__((always_inline))
uint quantize_code_word(thread uint& control_word, thread int& quant)
{
    if ((quant != 0) && (control_word != 0u))
    {
        int q_bits = int(extract_bits(control_word, uint(16), uint(4)));
        int sub_quant = min(q_bits, quant);
        q_bits -= sub_quant;
        quant -= sub_quant;
        if (quant != 0)
        {
            quant = min(quant, 3);
            uint plane0 = control_word & 21845u;
            uint plane1 = (control_word & 43690u) >> uint(1);
            uint plane2 = plane0 & plane1;
            do
            {
                plane0 = plane1;
                plane1 = plane2;
                plane2 = 0u;
                quant--;
            } while (quant != 0);
            plane0 &= (~plane1);
            uint new_control_word = plane0 | (plane1 << uint(1));
            control_word = insert_bits(control_word, new_control_word, uint(0), uint(16));
        }
        control_word = insert_bits(control_word, uint(q_bits), uint(16), uint(4));
    }
    return control_word;
}

static inline __attribute__((always_inline))
uint compute_required_8x8_size(thread const uint& control_word)
{
    int q_bits = int(extract_bits(control_word, uint(16), uint(4)));
    uint lsbs = control_word & 21845u;
    uint msbs = control_word & 43690u;
    uint msbs_shift = msbs >> uint(1);
    msbs |= msbs_shift;
    return uint((int(popcount(lsbs)) + int(popcount(msbs))) + (q_bits * 8));
}

static inline __attribute__((always_inline))
uint modify_quant_code(thread uint& code, thread const int& quant)
{
    int e = int(extract_bits(code, uint(3), uint(5)));
    e = max((e - quant), 0);
    code = insert_bits(code, uint(e), uint(3), uint(5));
    return code;
}

static inline __attribute__((always_inline))
uint inclusive_add_clustered16(thread uint& v, thread uint& gl_SubgroupInvocationID)
{
    for (uint i = 1u; i < 16u; i *= 2u)
    {
        uint up = spvSubgroupShuffleUp(v, i);
        v += (((gl_SubgroupInvocationID & 15u) >= i) ? up : 0u);
    }
    return v;
}

static inline __attribute__((always_inline))
uint copy_bytes(thread uint& output_offset, thread uint& input_offset, thread uint& count, device Payloads& payload_data, device BitstreamPayload8Bit& bitstream_data_8b)
{
    uint significant_mask = 0u;
    do
    {
        uint in_data = uint(payload_data.data[input_offset]);
        significant_mask |= in_data;
        uint _174 = output_offset;
        output_offset = _174 + uint(1);
        bitstream_data_8b.data[_174] = uchar(in_data);
        count--;
        input_offset++;
    } while (count > 0u);
    return significant_mask;
}

static inline __attribute__((always_inline))
void append_sign_plane(thread const uint& bank, thread uint& local_sign_offset, thread const uint& sign_mask, thread uint& significant_mask, thread uint& pending_sign_write, thread uint& pending_sign_mask, threadgroup spvUnsafeArray<spvUnsafeArray<uint, 32>, 4>& shared_sign_bank)
{
    while (significant_mask != 0u)
    {
        int bit = int(spvFindLSB(significant_mask));
        significant_mask &= (significant_mask - 1u);
        int out_bit = int(local_sign_offset & 31u);
        pending_sign_write = insert_bits(pending_sign_write, extract_bits(sign_mask, uint(bit), uint(1)), uint(out_bit), uint(1));
        pending_sign_mask = insert_bits(pending_sign_mask, 1u, uint(out_bit), uint(1));
        if (out_bit == 31)
        {
            if (pending_sign_mask == 4294967295u)
            {
                shared_sign_bank[bank][local_sign_offset / 32u] = pending_sign_write;
            }
            else
            {
                uint _293 = atomic_fetch_and_explicit((threadgroup atomic_uint*)&shared_sign_bank[bank][local_sign_offset / 32u], ~pending_sign_mask, memory_order_relaxed);
                uint _301 = atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_sign_bank[bank][local_sign_offset / 32u], pending_sign_write & pending_sign_mask, memory_order_relaxed);
            }
            pending_sign_mask = 0u;
        }
        local_sign_offset++;
    }
}

static inline __attribute__((always_inline))
void flush_sign_plane(thread const uint& bank, thread const uint& local_sign_offset, thread uint& pending_sign_write, thread uint& pending_sign_mask, threadgroup spvUnsafeArray<spvUnsafeArray<uint, 32>, 4>& shared_sign_bank)
{
    if (pending_sign_mask != 0u)
    {
        uint _314 = atomic_fetch_and_explicit((threadgroup atomic_uint*)&shared_sign_bank[bank][local_sign_offset / 32u], ~pending_sign_mask, memory_order_relaxed);
        uint _322 = atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_sign_bank[bank][local_sign_offset / 32u], pending_sign_write & pending_sign_mask, memory_order_relaxed);
        pending_sign_mask = 0u;
    }
}

kernel void pyrowave_block_packing(device Payloads& payload_data [[buffer(0)]], device void* spvBufferAliasSet0Binding0 [[buffer(1)]], constant Registers& registers [[buffer(2)]], device RateControlQuant& quant_data [[buffer(3)]], device SSBOMeta& block_meta [[buffer(4)]], device SSBOBlockStats& block_stats [[buffer(5)]], device BitstreamMeta& bitstream_meta [[buffer(6)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]], uint gl_SubgroupSize [[threads_per_simdgroup]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    device auto& bitstream_data_8b = *(device BitstreamPayload8Bit*)spvBufferAliasSet0Binding0;
    device auto& bitstream_data = *(device BitstreamPayload*)spvBufferAliasSet0Binding0;
    device auto& bitstream_data_16b = *(device BitstreamPayload16Bit*)spvBufferAliasSet0Binding0;
    threadgroup spvUnsafeArray<spvUnsafeArray<uint, 32>, 4> shared_sign_bank;
    uint pending_sign_write = 0u;
    uint pending_sign_mask = 0u;
    uint index = gl_SubgroupInvocationID + (gl_SubgroupSize * gl_SubgroupID);
    uint linear_block_32x32_index = index >> uint(4);
    int2 block32x32_index = int2(2) * int2(gl_WorkGroupID.xy);
    block32x32_index.x += int(extract_bits(index, uint(4), uint(1)));
    block32x32_index.y += int(extract_bits(index, uint(5), uint(1)));
    int2 local_block_index = int2(int(extract_bits(index, uint(0), uint(2))), int(extract_bits(index, uint(2), uint(2))));
    int2 block8x8_index = (int2(4) * block32x32_index) + local_block_index;
    bool in_range_8x8 = all(block8x8_index < registers.resolution_8x8_blocks);
    bool in_range_32x32 = all(block32x32_index < registers.resolution_32x32_blocks);
    uint num_bits_for_q = 0u;
    int quant;
    if (in_range_32x32)
    {
        int block_index = (registers.block_offset_32x32 + (registers.block_stride_32x32 * block32x32_index.y)) + block32x32_index.x;
        quant = quant_data.data[block_index];
    }
    else
    {
        quant = 0;
    }
    BlockMeta meta;
    if (in_range_8x8)
    {
        int block_index_1 = (registers.block_offset_8x8 + (registers.block_stride_8x8 * block8x8_index.y)) + block8x8_index.x;
        BlockMeta _449;
        _449.code_word = block_meta.meta[block_index_1].code_word;
        _449.offset = block_meta.meta[block_index_1].offset;
        meta = _449;
        uint num_planes = block_stats.stats[block_index_1].num_planes;
        num_bits_for_q = uint(block_stats.stats[block_index_1].errors[min(num_planes, uint(quant))].payload_cost);
    }
    else
    {
        meta = BlockMeta{ 0u, 0u };
    }
    uint param = meta.code_word;
    int param_1 = quant;
    uint _481 = quantize_code_word(param, param_1);
    uint code_word = _481;
    bool active_code_word = (code_word & 65535u) != 0u;
    uint4 code_word_ballot = spvSubgroupBallot(active_code_word);
    uint _499;
    if ((gl_SubgroupSize >= 64u) && (linear_block_32x32_index >= 2u))
    {
        _499 = code_word_ballot.y;
    }
    else
    {
        _499 = code_word_ballot.x;
    }
    uint local_ballot = _499;
    local_ballot = extract_bits(local_ballot, uint(int(16u * (linear_block_32x32_index & 1u))), uint(16));
    uint param_2 = code_word;
    uint required_plane_bytes = compute_required_8x8_size(param_2);
    uint required_sign_bits = num_bits_for_q - (required_plane_bytes * 8u);
    uint required_bits_with_meta = num_bits_for_q;
    if (required_bits_with_meta != 0u)
    {
        required_bits_with_meta += 24u;
    }
    bool _538 = all(block32x32_index < registers.resolution_32x32_blocks);
    bool _544;
    if (_538)
    {
        _544 = (index & 15u) == 15u;
    }
    else
    {
        _544 = _538;
    }
    bool writes_header = _544;
    uint payload_total_bits = spvClustered_sum<16>(required_bits_with_meta, gl_SubgroupInvocationID);
    uint payload_total_words = (payload_total_bits + 31u) / 32u;
    if (payload_total_words != 0u)
    {
        payload_total_words += 2u;
    }
    uint global_payload_offset = 0u;
    if (writes_header && (payload_total_words != 0u))
    {
        uint _567 = atomic_fetch_add_explicit((device atomic_uint*)&payload_data.bitstream_payload_counter, payload_total_words, memory_order_relaxed);
        global_payload_offset = _567;
    }
    global_payload_offset = spvSubgroupShuffle(global_payload_offset, gl_SubgroupInvocationID | 15u);
    if (writes_header)
    {
        uint block_index_2 = uint((registers.block_offset_32x32 + (block32x32_index.y * registers.block_stride_32x32)) + block32x32_index.x);
        if (payload_total_words != 0u)
        {
            bitstream_data.data[global_payload_offset + 0u] = (local_ballot | (payload_total_words << uint(16))) | (registers.sequence_code << uint(28));
            uint param_3 = registers.quant_resolution_code;
            int param_4 = quant;
            uint _616 = modify_quant_code(param_3, param_4);
            bitstream_data.data[global_payload_offset + 1u] = _616 | (block_index_2 << uint(8));
        }
        BitstreamPacket_1 _630 = BitstreamPacket_1{ global_payload_offset, payload_total_words };
        BitstreamPacket _633;
        _633.offset = _630.offset;
        _633.num_words = _630.num_words;
        bitstream_meta.packets[block_index_2] = _633;
    }
    uint total_subblocks = uint(int(popcount(local_ballot)));
    uint param_5 = required_sign_bits;
    uint _641 = inclusive_add_clustered16(param_5, gl_SubgroupInvocationID);
    uint total_sign_bits = _641;
    uint param_6 = required_plane_bytes;
    uint _645 = inclusive_add_clustered16(param_6, gl_SubgroupInvocationID);
    uint local_planes_offset = _645 - required_plane_bytes;
    uint local_sign_offset = total_sign_bits - required_sign_bits;
    uint global_planes_offset = ((4u * global_payload_offset) + (3u * total_subblocks)) + 8u;
    uint global_sign_offset = global_planes_offset + spvClustered_sum<16>(required_plane_bytes, gl_SubgroupInvocationID);
    global_planes_offset += local_planes_offset;
    uint total_sign_bytes = (spvSubgroupShuffle(total_sign_bits, gl_SubgroupInvocationID | 15u) + 7u) / 8u;
    if (active_code_word)
    {
        uint block_header_offset = uint(int(popcount(extract_bits(local_ballot, uint(0), uint((local_block_index.y * 4) + local_block_index.x)))));
        uint in_q_bits = extract_bits(meta.code_word, uint(16), uint(4));
        uint out_q_bits = extract_bits(code_word, uint(16), uint(4));
        uint input_offset = meta.offset;
        uint output_offset = global_planes_offset;
        for (int bit_offset = 0; bit_offset < 16; bit_offset += 2)
        {
            uint out_planes = extract_bits(code_word, uint(bit_offset), uint(2)) + out_q_bits;
            uint in_planes = extract_bits(meta.code_word, uint(bit_offset), uint(2)) + in_q_bits;
            if (in_planes != 0u)
            {
                in_planes++;
            }
            uint sign_plane = uint(payload_data.data[input_offset]);
            if (out_planes != 0u)
            {
                uint param_7 = output_offset;
                uint param_8 = input_offset + 1u;
                uint param_9 = out_planes;
                uint _745 = copy_bytes(param_7, param_8, param_9, payload_data, bitstream_data_8b);
                output_offset = param_7;
                uint significant_mask = _745;
                uint param_10 = linear_block_32x32_index;
                uint param_11 = local_sign_offset;
                uint param_12 = sign_plane;
                uint param_13 = significant_mask;
                append_sign_plane(param_10, param_11, param_12, param_13, pending_sign_write, pending_sign_mask, shared_sign_bank);
                local_sign_offset = param_11;
            }
            input_offset += in_planes;
        }
        uint param_14 = linear_block_32x32_index;
        uint param_15 = local_sign_offset;
        flush_sign_plane(param_14, param_15, pending_sign_write, pending_sign_mask, shared_sign_bank);
        bitstream_data_16b.data[((2u * global_payload_offset) + block_header_offset) + 4u] = ushort(code_word);
        bitstream_data_8b.data[(((4u * global_payload_offset) + (2u * total_subblocks)) + block_header_offset) + 8u] = uchar(code_word >> uint(16));
    }
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture);
    uint _794 = index & 15u;
    for (uint i = _794; i < (total_sign_bytes / 4u); i += 16u)
    {
        uint sign_word = shared_sign_bank[linear_block_32x32_index][i];
        uint offset_8b = global_sign_offset + (4u * i);
        bitstream_data_8b.data[offset_8b + 0u] = uchar(sign_word >> uint(0));
        bitstream_data_8b.data[offset_8b + 1u] = uchar(sign_word >> uint(8));
        bitstream_data_8b.data[offset_8b + 2u] = uchar(sign_word >> uint(16));
        bitstream_data_8b.data[offset_8b + 3u] = uchar(sign_word >> uint(24));
    }
    uint _847 = (total_sign_bytes & 4294967292u) + (index & 15u);
    for (uint i_1 = _847; i_1 < total_sign_bytes; i_1 += 16u)
    {
        uint sign_word_1 = shared_sign_bank[linear_block_32x32_index][i_1 / 4u];
        uint offset_8b_1 = global_sign_offset + i_1;
        bitstream_data_8b.data[offset_8b_1] = uchar(sign_word_1 >> (8u * (i_1 & 3u)));
    }
}

