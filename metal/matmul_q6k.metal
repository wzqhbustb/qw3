// Q6_K vector × matrix: y = x @ W^T
// W is stored in GGUF Q6_K format: 256 weights per 210-byte super-block.

#include <metal_stdlib>
using namespace metal;

#define Q6K_BLOCK_SIZE 256

struct ds3_block_q6_K {
    uint8_t  ql[128];   /* lower 4 bits of the 256 quants */
    uint8_t  qh[64];    /* upper 2 bits of the 256 quants */
    int8_t   scales[16];/* one scale per 16-weight group */
    uint16_t d;         /* super-block scale (FP16) */
};

struct ds3_matmul_q6k_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_stride;    // bytes per row of W
    uint64_t weight_offset; // byte offset to start of matrix in W
};

static inline float q6k_scale_from_u16(uint16_t u)
{
    return float( as_type<half>(u) );
}

static inline int q6k_lanes_per_row_for_simd(int n)
{
    if (n >= 32) return 32;
    if (n >= 16) return 16;
    if (n >=  8) return  8;
    if (n >=  4) return  4;
    if (n >=  2) return  2;
    return 1;
}

static inline float block_dot_q6k(
    device const ds3_block_q6_K *blk,
    device const float *x)
{
    const float d = q6k_scale_from_u16(blk->d);
    device const uint8_t *ql = blk->ql;
    device const uint8_t *qh = blk->qh;
    device const int8_t  *sc = blk->scales;

    float sum = 0.0f;
    for (int g = 0; g < 2; ++g) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;

            const int8_t q1 = (int8_t)(((ql[l + 0] & 0x0F)      ) | ((qh[l] & 0x03) << 4)) - 32;
            const int8_t q2 = (int8_t)(((ql[l + 32] & 0x0F)     ) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
            const int8_t q3 = (int8_t)(((ql[l + 0] >> 4)        ) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
            const int8_t q4 = (int8_t)(((ql[l + 32] >> 4)       ) | (((qh[l] >> 6) & 0x03) << 4)) - 32;

            const float v1 = d * float(sc[is + 0]) * float(q1);
            const float v2 = d * float(sc[is + 2]) * float(q2);
            const float v3 = d * float(sc[is + 4]) * float(q3);
            const float v4 = d * float(sc[is + 6]) * float(q4);

            const int base = g * 128 + l;
            sum += x[base +  0] * v1;
            sum += x[base + 32] * v2;
            sum += x[base + 64] * v3;
            sum += x[base + 96] * v4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return sum;
}

kernel void matmul_vec_q6k(
    device const float        * x     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_q6k_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row = tgpig.x * ntg.x + tpitg.x;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q6_K *blocks = (device const ds3_block_q6_K *)wrow;

    const uint n_blocks = args.in_dim / Q6K_BLOCK_SIZE;
    float sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q6k(&blocks[b], &x[b * Q6K_BLOCK_SIZE]);
    }
    y[row] = sum;
}

// SIMD-group parallel reduction variant.
// One SIMD group processes one or more rows. Lanes split the Q6_K blocks of
// each row and reduce via simd_shuffle_down.
kernel void matmul_vec_q6k_simd(
    device const float        * x     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * y     [[buffer(2)]],
    constant ds3_matmul_q6k_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.in_dim / Q6K_BLOCK_SIZE);
    const int lanes_per_row = q6k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.out_dim) return;

    device const char * wrow = W + args.weight_offset + row * args.row_stride;
    device const ds3_block_q6_K *blocks = (device const ds3_block_q6_K *)wrow;

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q6k(&blocks[b], &x[b * Q6K_BLOCK_SIZE]);
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
 *   W: [N][K]  Q6_K
 *   C: [M][N]  FP32
 */
struct ds3_matmul_q6k_batch_args {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint64_t a_stride;
    uint64_t c_stride;
    uint64_t weight_row_stride;
    uint64_t weight_offset;
};

kernel void matmul_q6k(
    device const float        * A     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * C     [[buffer(2)]],
    constant ds3_matmul_q6k_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint m = tgpig.x * ntg.x + tpitg.x;
    const uint n = tgpig.y * ntg.y + tpitg.y;
    if (m >= args.M || n >= args.N) return;

    device const char * wrow = W + args.weight_offset + n * args.weight_row_stride;
    device const ds3_block_q6_K *blocks = (device const ds3_block_q6_K *)wrow;

    const uint n_blocks = args.K / Q6K_BLOCK_SIZE;
    float sum = 0.0f;
    device const float *a_row = A + m * args.a_stride;
    for (uint b = 0; b < n_blocks; b++) {
        sum += block_dot_q6k(&blocks[b], &a_row[b * Q6K_BLOCK_SIZE]);
    }
    C[m * args.c_stride + n] = sum;
}

kernel void matmul_q6k_batch_simd(
    device const float        * A     [[buffer(0)]],
    device const char         * W     [[buffer(1)]],
    device       float        * C     [[buffer(2)]],
    constant ds3_matmul_q6k_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint m = tgpig.x;
    const int n_blocks      = int(args.K / Q6K_BLOCK_SIZE);
    const int lanes_per_row = q6k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tpitg.x);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int n = int(tgpig.y) * rows_per_simd + row_in_simd;
    if (m >= args.M || n >= args.N) return;

    device const char * wrow = W + args.weight_offset + n * args.weight_row_stride;
    device const ds3_block_q6_K *blocks = (device const ds3_block_q6_K *)wrow;
    device const float *a_row = A + m * args.a_stride;

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q6k(&blocks[b], &a_row[b * Q6K_BLOCK_SIZE]);
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
