// Lightweight FP32 elementwise kernels used by the engine graph.
// These stay on the GPU to avoid small host<->device transfers.

#include <metal_stdlib>
using namespace metal;

kernel void vec_copy_f32(device const float *src [[buffer(0)]],
                         device       float *dst [[buffer(1)]],
                         constant     uint  &n   [[buffer(2)]],
                         uint idx [[thread_position_in_grid]])
{
    if (idx < n) dst[idx] = src[idx];
}

kernel void vec_zero_f32(device float *dst [[buffer(0)]],
                         constant uint &n [[buffer(1)]],
                         uint idx [[thread_position_in_grid]])
{
    if (idx < n) dst[idx] = 0.0f;
}

// b[i] += a[i]
kernel void vec_add_f32(device const float *a [[buffer(0)]],
                        device       float *b [[buffer(1)]],
                        constant     uint  &n [[buffer(2)]],
                        uint idx [[thread_position_in_grid]])
{
    if (idx < n) b[idx] += a[idx];
}

// c[i] = a[i] + b[i]
kernel void vec_add3_f32(device const float *a [[buffer(0)]],
                         device const float *b [[buffer(1)]],
                         device       float *c [[buffer(2)]],
                         constant     uint  &n [[buffer(3)]],
                         uint idx [[thread_position_in_grid]])
{
    if (idx < n) c[idx] = a[idx] + b[idx];
}

/* Gather rows: dst[i][j] = src[ indices[offset + i] ][j]
 *   src: [n_rows][n_cols]
 *   indices: list of row indices (int32)
 *   dst: [count][n_cols]
 * One thread per (i, j) element. */
struct ds3_gather_args {
    uint32_t n_cols;
    uint32_t count;
    uint32_t offset;
};

kernel void gather_rows_f32(device const float        *src   [[buffer(0)]],
                            device const int32_t      *ids   [[buffer(1)]],
                            device       float        *dst   [[buffer(2)]],
                            constant ds3_gather_args &args  [[buffer(3)]],
                            uint idx [[thread_position_in_grid]])
{
    const uint i = idx / args.n_cols;
    const uint j = idx % args.n_cols;
    if (i >= args.count) return;
    const uint src_row = (uint)ids[args.offset + i];
    dst[i * args.n_cols + j] = src[src_row * args.n_cols + j];
}

/* Scatter-add weighted: dst[ ids[offset + i] ][j] += scores[offset + i] * src[i][j]
 *   src: [count][n_cols]
 *   dst: [n_rows][n_cols]
 * One thread per (i, j) element. */
struct ds3_scatter_args {
    uint32_t n_cols;
    uint32_t count;
    uint32_t offset;
};

kernel void scatter_add_weighted_f32(device const float          *src    [[buffer(0)]],
                                     device const float          *scores [[buffer(1)]],
                                     device const int32_t        *ids    [[buffer(2)]],
                                     device       float          *dst    [[buffer(3)]],
                                     constant ds3_scatter_args &args   [[buffer(4)]],
                                     uint idx [[thread_position_in_grid]])
{
    const uint i = idx / args.n_cols;
    const uint j = idx % args.n_cols;
    if (i >= args.count) return;
    const uint dst_row = (uint)ids[args.offset + i];
    const float w = scores[args.offset + i];
    dst[dst_row * args.n_cols + j] += w * src[i * args.n_cols + j];
}
