// Q4_K vector × matrix: y = x @ W^T
// W is stored in GGUF Q4_K format: 256 weights per 144-byte super-block.

#include <metal_stdlib>
using namespace metal;

#define Q4K_BLOCK_SIZE 256
#define Q4K_SCALE_SIZE 12

struct ds3_block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[Q4K_SCALE_SIZE];
    uint8_t  qs[Q4K_BLOCK_SIZE / 2];
};

struct ds3_matmul_q4k_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_stride; // bytes per row of W
    uint64_t weight_offset; // byte offset to start of matrix in W
};

static inline void get_scale_min_k4(int j, const device uint8_t *q, thread uint8_t *d, thread uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Reinterpret the raw uint16 bits stored in GGUF as an IEEE-754 half-precision
// float. Metal's as_type<half> does exactly this bit-cast, matching the CPU-side
// memcpy(_Float16) trick in the reference dequantizer.
static inline float q4k_scale_from_u16(uint16_t u)
{
    return float( as_type<half>(u) );
}

static inline int q4k_lanes_per_row_for_simd(int n)
{
    if (n >= 32) return 32;
    if (n >= 16) return 16;
    if (n >=  8) return  8;
    if (n >=  4) return  4;
    if (n >=  2) return  2;
    return 1;
}

static inline float block_dot_q4k(
    device const ds3_block_q4_K *blk,
    device const float *x)
{
    const device uint8_t *quants = blk->qs;
    const float d   = q4k_scale_from_u16(blk->d);
    const float min = q4k_scale_from_u16(blk->dmin);

    float sum = 0.0f;
    int is = 0;
    uint8_t sc, m;

    for (int j = 0; j < Q4K_BLOCK_SIZE; j += 64) {
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;

        for (int l = 0; l < 32; ++l) sum += (d1 * (quants[l] & 0x0F) - m1) * x[l];
        for (int l = 0; l < 32; ++l) sum += (d2 * (quants[l] >>   4) - m2) * x[l + 32];

        quants += 32;
        x += 64;
        is += 2;
    }
    return sum;
}

static inline float block_dot_q4k_half(
    device const ds3_block_q4_K *blk,
    device const half *x)
{
    const device uint8_t *quants = blk->qs;
    const float d   = q4k_scale_from_u16(blk->d);
    const float min = q4k_scale_from_u16(blk->dmin);

    float sum = 0.0f;
    int is = 0;
    uint8_t sc, m;

    for (int j = 0; j < Q4K_BLOCK_SIZE; j += 64) {
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;

        for (int l = 0; l < 32; ++l) sum += (d1 * (quants[l] & 0x0F) - m1) * float(x[l]);
        for (int l = 0; l < 32; ++l) sum += (d2 * (quants[l] >>   4) - m2) * float(x[l + 32]);

        quants += 32;
        x += 64;
        is += 2;
    }
    return sum;
}

kernel void matmul_vec_q4k(
    device const float        * x     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_q4k_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q4_K *blocks = (device const ds3_block_q4_K *)wrow;

    const uint n_blocks = args.in_dim / Q4K_BLOCK_SIZE;
    float sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q4k(&blocks[b], &x[b * Q4K_BLOCK_SIZE]);
    }
    y[row] = sum;
}

// SIMD-group parallel reduction variant.
// One SIMD group processes one or more rows. Lanes within the group split the
// Q4_K blocks of each row, then reduce via simd_shuffle_down.
// The host must use the same lanes_per_row_for_simd(in_dim/256) rule to compute
// the grid size.
kernel void matmul_vec_q4k_simd(
    device const float        * x     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_q4k_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.in_dim / Q4K_BLOCK_SIZE);
    const int lanes_per_row = q4k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q4_K *blocks = (device const ds3_block_q4_K *)wrow;

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q4k(&blocks[b], &x[b * Q4K_BLOCK_SIZE]);
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

// Half-precision input variant (e.g. for FP16 KV cache activations).
kernel void matmul_vec_q4k_half(
    device const half         * x     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_q4k_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q4_K *blocks = (device const ds3_block_q4_K *)wrow;

    const uint n_blocks = args.in_dim / Q4K_BLOCK_SIZE;
    float sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q4k_half(&blocks[b], &x[b * Q4K_BLOCK_SIZE]);
    }
    y[row] = sum;
}

/* Batched matrix × quantized matrix: C = A @ W^T
 *   A: [M][K]  FP32, row-major
 *   W: [N][K]  Q4_K, row-major (each row has in_dim/K_Q4K_BLOCK_SIZE super-blocks)
 *   C: [M][N]  FP32, row-major
 */
struct ds3_matmul_q4k_batch_args {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint64_t a_stride;       // elements per row of A (== K)
    uint64_t c_stride;       // elements per row of C (== N)
    uint64_t weight_row_stride; // bytes per row of W
    uint64_t weight_offset;  // byte offset to start of matrix in W
};

kernel void matmul_q4k(
    device const float        * A     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * C     [[buffer(2)]],
    constant ds3_matmul_q4k_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint m = tgpig.x * ntg.x + tpitg.x;
    const uint n = tgpig.y * ntg.y + tpitg.y;
    if (m >= args.M || n >= args.N) return;

    device const char * wrow = W + args.weight_offset + n * args.weight_row_stride;
    device const ds3_block_q4_K *blocks = (device const ds3_block_q4_K *)wrow;

    const uint n_blocks = args.K / Q4K_BLOCK_SIZE;
    float sum = 0.0f;
    device const float *a_row = A + m * args.a_stride;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q4k(&blocks[b], &a_row[b * Q4K_BLOCK_SIZE]);
    }
    C[m * args.c_stride + n] = sum;
}

/* SIMD-group batched variant.  One SIMD group computes one output element
 * (m,n): lanes split the K-axis blocks and reduce via simd_shuffle_down,
 * matching the summation order of matmul_vec_q4k_simd. */
kernel void matmul_q4k_batch_simd(
    device const float        * A     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * C     [[buffer(2)]],
    constant ds3_matmul_q4k_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint m = tgpig.x;
    const int n_blocks      = int(args.K / Q4K_BLOCK_SIZE);
    const int lanes_per_row = q4k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tpitg.x);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int n = int(tgpig.y) * rows_per_simd + row_in_simd;
    if (m >= args.M || n >= args.N) return;

    device const char * wrow = W + args.weight_offset + n * args.weight_row_stride;
    device const ds3_block_q4_K *blocks = (device const ds3_block_q4_K *)wrow;
    device const float *a_row = A + m * args.a_stride;

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q4k(&blocks[b], &a_row[b * Q4K_BLOCK_SIZE]);
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
