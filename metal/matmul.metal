// Simple FP32 matrix multiply: C = A × B
// A: [M][K], B: [K][N], C: [M][N]  (row-major)
// Each thread computes one element of C.

#include <metal_stdlib>
using namespace metal;

struct ds3_matmul_args {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint64_t lda; // row stride of A (in elements)
    uint64_t ldb; // row stride of B (in elements)
    uint64_t ldc; // row stride of C (in elements)
};

kernel void matmul_f32(
    device const float      * A     [[buffer(0)]],
    device const float      * B     [[buffer(1)]],
    device       float      * C     [[buffer(2)]],
    constant ds3_matmul_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row = tgpig.x * ntg.x + tpitg.x;
    const uint col = tgpig.y * ntg.y + tpitg.y;
    if (row >= args.M || col >= args.N) return;

    float sum = 0.0f;
    for (uint k = 0; k < args.K; k++) {
        sum += A[row * args.lda + k] * B[k * args.ldb + col];
    }
    C[row * args.ldc + col] = sum;
}

#define MM_TILE_SIZE 16

// Tiled FP32 matmul. Each threadgroup cooperatively loads tiles of A and B
// into threadgroup memory and computes a TILE_SIZE x TILE_SIZE block of C.
// For prefill-style shapes (M > 1, large N) this is much faster than the
// naive one-thread-per-element kernel above.
kernel void matmul_f32_tiled(
    device const float      * A     [[buffer(0)]],
    device const float      * B     [[buffer(1)]],
    device       float      * C     [[buffer(2)]],
    constant ds3_matmul_args & args [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    const uint row = gid.y;
    const uint col = gid.x;

    threadgroup float As[MM_TILE_SIZE][MM_TILE_SIZE];
    threadgroup float Bs[MM_TILE_SIZE][MM_TILE_SIZE];

    float sum = 0.0f;
    const uint tiles = (args.K + MM_TILE_SIZE - 1) / MM_TILE_SIZE;

    for (uint t = 0; t < tiles; t++) {
        const uint kA = t * MM_TILE_SIZE + lid.x;
        const uint kB = t * MM_TILE_SIZE + lid.y;

        if (row < args.M && kA < args.K) {
            As[lid.y][lid.x] = A[row * args.lda + kA];
        } else {
            As[lid.y][lid.x] = 0.0f;
        }

        if (kB < args.K && col < args.N) {
            Bs[lid.y][lid.x] = B[kB * args.ldb + col];
        } else {
            Bs[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < MM_TILE_SIZE; k++) {
            sum += As[lid.y][k] * Bs[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < args.M && col < args.N) {
        C[row * args.ldc + col] = sum;
    }
}
