#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

kernel void matmul_tiled(device const float *A      [[buffer(0)]],
                         device const float *B      [[buffer(1)]],
                         device float       *C      [[buffer(2)]],
                         constant uint      &M      [[buffer(3)]],
                         constant uint      &N      [[buffer(4)]],
                         constant uint      &K      [[buffer(5)]],
                         uint2 gid  [[thread_position_in_grid]],
                         uint2 lid  [[thread_position_in_threadgroup]])
{
    // gid = (col, row)  — which element of C this thread computes
    // lid = (tx, ty)    — where this thread sits inside its threadgroup

    uint row = gid.y;
    uint col = gid.x;

    // Early exit for threads outside the C matrix.
    // dispatchThreads handles grid-level masking, but for 2D we still
    // need to check because the grid might have extra rows/cols.
    if (row >= M || col >= N) return;

    // ── Threadgroup memory (shared memory) ──────────────────────
    // Every thread in the threadgroup can read/write these arrays.
    // Fast (~48KB total, ~1 TB/s bandwidth vs ~400 GB/s for device mem).
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // ── Slide across the K dimension, one tile at a time ─────────
    for (uint t = 0; t < tiles; t++) {

        // --- Cooperative load: each thread loads ONE element ---
        //     from A and ONE from B into threadgroup memory.
        uint kA = t * TILE_SIZE + lid.x;  // K-index into A
        uint kB = t * TILE_SIZE + lid.y;  // K-index into B

        if (row < M && kA < K) {
            As[lid.y][lid.x] = A[row * K + kA];
        } else {
            As[lid.y][lid.x] = 0.0f;
        }

        if (kB < K && col < N) {
            Bs[lid.y][lid.x] = B[kB * N + col];
        } else {
            Bs[lid.y][lid.x] = 0.0f;
        }

        // --- Barrier: all threads must finish loading before ---
        //     anyone starts computing with this tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute partial dot product using fast memory ---
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[lid.y][k] * Bs[k][lid.x];
            //       ↑ A[row][k_tile]   ↑ B[k_tile][col]
        }

        // --- Barrier: all threads must finish computing before ---
        //     anyone starts loading the NEXT tile (overwriting As/Bs).
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = sum;
}
