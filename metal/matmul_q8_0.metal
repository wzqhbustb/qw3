// Q8_0 vector × matrix: y = x @ W^T
// W is stored in GGUF Q8_0 format: 32 weights per 34-byte block.

#include <metal_stdlib>
using namespace metal;

#define Q8_0_BLOCK_SIZE 32

struct ds3_block_q8_0 {
    uint16_t d;
    int8_t   qs[Q8_0_BLOCK_SIZE];
};

struct ds3_matmul_q8_0_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_stride;    // bytes per row of W
    uint64_t weight_offset; // byte offset to start of matrix in W
};

// Reinterpret the raw uint16 bits in blk->d as an IEEE-754 half-precision float.
// This matches the CPU reference's memcpy(_Float16) decode.
static inline float q8_0_scale_from_u16(uint16_t u)
{
    return float( as_type<half>(u) );
}

static inline int q8_0_lanes_per_row_for_simd(int n)
{
    if (n >= 32) return 32;
    if (n >= 16) return 16;
    if (n >=  8) return  8;
    if (n >=  4) return  4;
    if (n >=  2) return  2;
    return 1;
}

static inline float block_dot_q8_0(
    device const ds3_block_q8_0 *blk,
    device const float *x)
{
    const float d = q8_0_scale_from_u16(blk->d);
    float sum = 0.0f;
    for (int i = 0; i < Q8_0_BLOCK_SIZE; i++) {
        sum += d * float(blk->qs[i]) * x[i];
    }
    return sum;
}

kernel void matmul_vec_q8_0(
    device const float          * x     [[buffer(0)]],
    device const char           * W     [[buffer(1)]],
    device       float          * y     [[buffer(2)]],
    constant ds3_matmul_q8_0_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q8_0 *blocks = (device const ds3_block_q8_0 *)wrow;

    const uint n_blocks = args.in_dim / Q8_0_BLOCK_SIZE;
    float sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q8_0(&blocks[b], &x[b * Q8_0_BLOCK_SIZE]);
    }
    y[row] = sum;
}

// SIMD-group parallel reduction variant. Because Q8_0 blocks are small, in_dim
// usually yields many blocks per row; each lane therefore takes multiple blocks.
// The host must use the same lanes_per_row_for_simd(in_dim/32) rule to compute
// the grid size.
kernel void matmul_vec_q8_0_simd(
    device const float          * x     [[buffer(0)]],
    device const char           * W     [[buffer(1)]],
    device       float          * y     [[buffer(2)]],
    constant ds3_matmul_q8_0_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.in_dim / Q8_0_BLOCK_SIZE);
    const int lanes_per_row = q8_0_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q8_0 *blocks = (device const ds3_block_q8_0 *)wrow;

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q8_0(&blocks[b], &x[b * Q8_0_BLOCK_SIZE]);
    }

    float sum = partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            sum += simd_shuffle_down(sum, offset);
        }
    }

    if (block_in_row == 0) {
        y[row] = sum;
    }
}

/* Batched matrix × quantized matrix: C = A @ W^T
 *   A: [M][K]  FP32
 *   W: [N][K]  Q8_0
 *   C: [M][N]  FP32
 */
struct ds3_matmul_q8_0_batch_args {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint64_t a_stride;
    uint64_t c_stride;
    uint64_t weight_row_stride;
    uint64_t weight_offset;
};

kernel void matmul_q8_0(
    device const float          * A     [[buffer(0)]],
    device const char           * W     [[buffer(1)]],
    device       float          * C     [[buffer(2)]],
    constant ds3_matmul_q8_0_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint m = tgpig.x * ntg.x + tpitg.x;
    const uint n = tgpig.y * ntg.y + tpitg.y;
    if (m >= args.M || n >= args.N) return;

    device const char * wrow = W + args.weight_offset + n * args.weight_row_stride;
    device const ds3_block_q8_0 *blocks = (device const ds3_block_q8_0 *)wrow;

    const uint n_blocks = args.K / Q8_0_BLOCK_SIZE;
    float sum = 0.0f;
    device const float *a_row = A + m * args.a_stride;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q8_0(&blocks[b], &a_row[b * Q8_0_BLOCK_SIZE]);
    }
    C[m * args.c_stride + n] = sum;
}

kernel void matmul_q8_0_batch_simd(
    device const float          * A     [[buffer(0)]],
    device const char           * W     [[buffer(1)]],
    device       float          * C     [[buffer(2)]],
    constant ds3_matmul_q8_0_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint m = tgpig.x;
    const int n_blocks      = int(args.K / Q8_0_BLOCK_SIZE);
    const int lanes_per_row = q8_0_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tpitg.x);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int n = int(tgpig.y) * rows_per_simd + row_in_simd;
    if (m >= args.M || n >= args.N) return;

    device const char * wrow = W + args.weight_offset + n * args.weight_row_stride;
    device const ds3_block_q8_0 *blocks = (device const ds3_block_q8_0 *)wrow;
    device const float *a_row = A + m * args.a_stride;

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q8_0(&blocks[b], &a_row[b * Q8_0_BLOCK_SIZE]);
    }

    float sum = partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            sum += simd_shuffle_down(sum, offset);
        }
    }

    if (block_in_row == 0) {
        C[m * args.c_stride + n] = sum;
    }
}
