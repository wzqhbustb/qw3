// FP32 vector × matrix: y = x @ W^T
// x: [in_dim], W: [out_dim][in_dim] row-major, y: [out_dim]

#include <metal_stdlib>
using namespace metal;

struct ds3_matmul_vec_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_stride; // bytes per row of W
};

// For SIMD-group reduction: choose a divisor of 32 that is <= n.
// This guarantees an integer number of rows per SIMD group.
static inline int f32_lanes_per_row_for_simd(int n)
{
    if (n >= 32) return 32;
    if (n >= 16) return 16;
    if (n >=  8) return  8;
    if (n >=  4) return  4;
    if (n >=  2) return  2;
    return 1;
}

kernel void matmul_vec_f32(
    device const float        * x     [[buffer(0)]],
    device const float        * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_vec_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.out_dim) return;

    device const float * wrow = (device const float *)((device const char *)W + row * args.row_stride);

    float sum = 0.0f;
    for (uint i = 0; i < args.in_dim; i++) {
        sum += x[i] * wrow[i];
    }
    y[row] = sum;
}

// SIMD-group parallel reduction variant. Lanes in a SIMD group split the
// dot-product columns of one (or more) rows, then reduce via simd_shuffle_down.
// The host dispatch must use the same lanes_per_row_for_simd() rule to compute
// the grid size.
kernel void matmul_vec_f32_simd(
    device const float        * x     [[buffer(0)]],
    device const float        * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_vec_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int in_dim        = int(args.in_dim);
    const int lanes_per_row = f32_lanes_per_row_for_simd(in_dim);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane        = int(tiisg);
    const int row_in_simd = lane / lanes_per_row;
    const int col_in_row  = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.out_dim) return;

    device const float * wrow = (device const float *)((device const char *)W + row * args.row_stride);

    float partial = 0.0f;
    for (int i = col_in_row; i < in_dim; i += lanes_per_row) {
        partial += x[i] * wrow[i];
    }

    float sum = partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (col_in_row + offset < lanes_per_row) {
            sum += simd_shuffle_down(sum, offset);
        }
    }

    if (col_in_row == 0) {
        y[row] = sum;
    }
}
