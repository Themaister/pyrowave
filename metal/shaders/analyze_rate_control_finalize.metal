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

struct Buckets
{
    char _m0_pad[64];
    uint4 total_savings_per_bucket[512];
};

kernel void pyrowave_analyze_rate_control_finalize(device Buckets& buckets [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]])
{
    threadgroup spvUnsafeArray<uint, 512> shared_scan;
    uint4 v = buckets.total_savings_per_bucket[gl_LocalInvocationIndex];
    v.y += v.x;
    v.z += v.y;
    v.w += v.z;
    shared_scan[gl_LocalInvocationIndex] = v.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint _step = 1u; _step < 256u; _step *= 2u)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint shuffled_up = 0u;
        if (gl_LocalInvocationIndex >= _step)
        {
            shuffled_up = shared_scan[gl_LocalInvocationIndex - _step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v += uint4(shuffled_up);
        shared_scan[gl_LocalInvocationIndex] = v.w;
    }
    buckets.total_savings_per_bucket[gl_LocalInvocationIndex] = v;
}

