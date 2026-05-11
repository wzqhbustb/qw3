// RMSNorm kernel for Qwen3.
// Computes: out = x / sqrt(mean(x^2) + eps) * weight
// Vectorized float4 loads, threadgroup reduction.

#include <metal_stdlib>
using namespace metal;

struct ds3_rms_norm_args {
    uint32_t n;          // elements per row
    uint32_t n_rows;     // number of rows (batch)
    float    eps;
    uint64_t src_stride; // byte stride between rows of input
    uint64_t dst_stride; // byte stride between rows of output
};

kernel void rms_norm_f32(
    device const float        * src      [[buffer(0)]],
    device const float        * weight   [[buffer(1)]],
    device       float        * dst      [[buffer(2)]],
    constant ds3_rms_norm_args & args    [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float         * shmem    [[threadgroup(0)]])
{
    const uint row = tgpig.x;
    if (row >= args.n_rows) return;

    device const float * x = (device const float *)((device const char *)src + row * args.src_stride);
    device       float * y = (device       float *)((device       char *)dst + row * args.dst_stride);

    // ---- sum of squares ----
    float sumf = 0.0f;
    const uint n_vec = args.n / 4;
    for (uint i = tpitg.x; i < n_vec; i += ntg.x) {
        float4 xv = ((device const float4 *)x)[i];
        sumf += dot(xv, xv);
    }
    // tail scalar elements
    for (uint i = tpitg.x + n_vec * 4; i < args.n; i += ntg.x) {
        float xv = x[i];
        sumf += xv * xv;
    }

    sumf = simd_sum(sumf);

    // cross-simdgroup reduction via threadgroup memory.
    // threadgroup memory is always allocated for at least 32 floats
    // (see ds3_metal.m), so shmem[tiisg] is always in bounds.
    if (sgitg == 0) shmem[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = simd_sum(shmem[tiisg]) / float(args.n);

    const float scale = 1.0f / sqrt(mean + args.eps);

    // ---- write output ----
    for (uint i = tpitg.x; i < n_vec; i += ntg.x) {
        float4 xv = ((device const float4 *)x)[i];
        float4 wv = ((device const float4 *)weight)[i];
        ((device float4 *)y)[i] = xv * scale * wv;
    }
    for (uint i = tpitg.x + n_vec * 4; i < args.n; i += ntg.x) {
        y[i] = x[i] * scale * weight[i];
    }
}
