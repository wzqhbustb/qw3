// GPT-NeoX style RoPE kernel for Qwen3.
// Qwen3 uses the NeoX rotation layout (not the adjacent-pair LLaMA layout):
//   x'_{i}            = x_{i}            * cos(m*theta_i) - x_{i+n/2} * sin(m*theta_i)
//   x'_{i + head_dim/2} = x_{i + head_dim/2} * cos(m*theta_i) + x_{i}     * sin(m*theta_i)
//   theta_i = theta_base ^ (-2i / head_dim)   for i = 0 .. head_dim/2 - 1

#include <metal_stdlib>
using namespace metal;

struct ds3_rope_args {
    uint32_t n_heads;       // number of heads
    uint32_t head_dim;      // dimension per head (must be even)
    uint32_t n_rows;        // batch size (seq_len)
    float    theta_base;    // e.g. 1e6 for Qwen3
    uint64_t src_stride;    // byte stride between rows
    uint64_t dst_stride;    // byte stride between rows
};

kernel void rope_f32(
    device const float      * src       [[buffer(0)]],
    device       float      * dst       [[buffer(1)]],
    constant ds3_rope_args  & args      [[buffer(2)]],
    device const int32_t    * positions [[buffer(3)]], // [n_rows] token positions
    device const float      * freq_table [[buffer(4)]], // [head_dim/2] precomputed theta_i
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint row   = tgpig.x;
    const uint head  = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;
    if (args.head_dim % 2 != 0) return;  /* odd head_dim not supported */

    const int pos = positions[row];

    device const float * x = (device const float *)((device const char *)src
                               + row * args.src_stride + head * args.head_dim * sizeof(float));
    device       float * y = (device       float *)((device       char *)dst
                               + row * args.dst_stride + head * args.head_dim * sizeof(float));

    const uint n_pairs = args.head_dim / 2;
    for (uint i = tpitg.x; i < n_pairs; i += ntg.x) {
        const float theta_i = freq_table[i];
        const float freq = float(pos) * theta_i;
        const float c = cos(freq);
        const float s = sin(freq);

        const float x0 = x[i];
        const float x1 = x[i + n_pairs];

        y[i]           = x0 * c - x1 * s;
        y[i + n_pairs] = x0 * s + x1 * c;
    }
}
