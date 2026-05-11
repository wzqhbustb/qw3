// Decode-only GQA attention kernels for Qwen3.
//
// Two-kernel design to avoid cross-threadgroup write-after-read races on the KV
// cache:
//   1. kv_cache_write   : writes the new k, v row into k_cache/v_cache at seq_pos.
//   2. attention_decode_* : reads q and the full cache, computes
//                           softmax(Q @ K^T) @ V.
//
// The host dispatches both kernels in the same command buffer; Metal guarantees
// that the second kernel sees all writes from the first.
//
// KV cache layout (tight, no padding):
//   [max_seq_len][n_kv_heads][head_dim]  FP16

#include <metal_stdlib>
using namespace metal;

#define MAX_HEAD_DIM 128

struct ds3_attention_args {
    uint32_t seq_pos;      // position of the new token
    uint32_t max_seq_len;  // allocated length of KV cache (currently tightly packed)
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
};

struct ds3_attention_batch_args {
    uint32_t seq_pos;      // starting position of the chunk in the KV cache
    uint32_t n_tokens;     // number of tokens in the chunk
    uint32_t max_seq_len;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
};

/* -------------------------------------------------------------------------- */
/* 1. KV cache write                                                          */
/* -------------------------------------------------------------------------- */

kernel void kv_cache_write(
    device const float        * k         [[buffer(0)]],
    device const float        * v         [[buffer(1)]],
    device       half         * k_cache   [[buffer(2)]],
    device       half         * v_cache   [[buffer(3)]],
    constant ds3_attention_args & args    [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint kvh = tgpig.x * ntg.x + tpitg.x;
    if (kvh >= args.n_kv_heads) return;

    const uint kv_stride = args.n_kv_heads * args.head_dim;
    const uint kv_base   = args.seq_pos * kv_stride + kvh * args.head_dim;

    for (uint d = 0; d < args.head_dim; ++d) {
        k_cache[kv_base + d] = half(k[kvh * args.head_dim + d]);
        v_cache[kv_base + d] = half(v[kvh * args.head_dim + d]);
    }
}

// Batched KV cache write for chunked prefill.
// Input k/v are [n_tokens][n_kv_heads][head_dim] FP32.
// Writes each token at cache position seq_pos + token_index.
kernel void kv_cache_write_batch(
    device const float        * k         [[buffer(0)]],
    device const float        * v         [[buffer(1)]],
    device       half         * k_cache   [[buffer(2)]],
    device       half         * v_cache   [[buffer(3)]],
    constant ds3_attention_batch_args & args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint t   = tgpig.x * ntg.x + tpitg.x;
    const uint kvh = tgpig.y * ntg.y + tpitg.y;
    if (t >= args.n_tokens || kvh >= args.n_kv_heads) return;

    const uint kv_stride = args.n_kv_heads * args.head_dim;
    const uint kv_base   = (args.seq_pos + t) * kv_stride + kvh * args.head_dim;
    const uint in_base   = (t * args.n_kv_heads + kvh) * args.head_dim;

    for (uint d = 0; d < args.head_dim; ++d) {
        k_cache[kv_base + d] = half(k[in_base + d]);
        v_cache[kv_base + d] = half(v[in_base + d]);
    }
}

/* -------------------------------------------------------------------------- */
/* 2. Baseline attention compute                                              */
/* -------------------------------------------------------------------------- */

kernel void attention_decode_gqa(
    device const float        * q         [[buffer(0)]],
    device const half         * k_cache   [[buffer(1)]],
    device const half         * v_cache   [[buffer(2)]],
    device       float        * output    [[buffer(3)]],
    constant ds3_attention_args & args    [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint qh = tgpig.x * ntg.x + tpitg.x;
    if (qh >= args.n_q_heads) return;

    const uint kvh          = qh / (args.n_q_heads / args.n_kv_heads);
    const uint seq_len      = args.seq_pos + 1;
    const float scale       = 1.0f / sqrt((float)args.head_dim);
    const uint kv_stride    = args.n_kv_heads * args.head_dim;
    const uint q_base       = qh * args.head_dim;

    /* Cache this Q head in private memory. */
    float q_head[MAX_HEAD_DIM];
    for (uint d = 0; d < args.head_dim; ++d) {
        q_head[d] = q[q_base + d];
    }

    /* First pass: compute max score for numerical stability. */
    float max_score = -FLT_MAX;
    for (uint pos = 0; pos < seq_len; ++pos) {
        const uint k_row = pos * kv_stride + kvh * args.head_dim;
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) {
            score += q_head[d] * float(k_cache[k_row + d]);
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    /* Second pass: softmax denominator and weighted sum of V. */
    float sum_exp = 0.0f;
    float out_head[MAX_HEAD_DIM];
    for (uint d = 0; d < args.head_dim; ++d) out_head[d] = 0.0f;

    for (uint pos = 0; pos < seq_len; ++pos) {
        const uint kv_row = pos * kv_stride + kvh * args.head_dim;
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) {
            score += q_head[d] * float(k_cache[kv_row + d]);
        }
        score *= scale;
        float w = exp(score - max_score);
        sum_exp += w;
        for (uint d = 0; d < args.head_dim; ++d) {
            out_head[d] += w * float(v_cache[kv_row + d]);
        }
    }

    const float inv_sum = 1.0f / sum_exp;
    for (uint d = 0; d < args.head_dim; ++d) {
        output[q_base + d] = out_head[d] * inv_sum;
    }
}

/* -------------------------------------------------------------------------- */
/* 3. SIMD-group parallel attention compute                                   */
/* -------------------------------------------------------------------------- */

kernel void attention_decode_gqa_simd(
    device const float        * q         [[buffer(0)]],
    device const half         * k_cache   [[buffer(1)]],
    device const half         * v_cache   [[buffer(2)]],
    device       float        * output    [[buffer(3)]],
    constant ds3_attention_args & args    [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const uint qh = tgpig.x;
    if (qh >= args.n_q_heads) return;

    const uint lane         = tiisg;
    const uint heads_per_kv = args.n_q_heads / args.n_kv_heads;
    const uint kvh          = qh / heads_per_kv;
    const uint seq_len      = args.seq_pos + 1;
    const float scale       = 1.0f / sqrt((float)args.head_dim);
    const uint kv_stride    = args.n_kv_heads * args.head_dim;
    const uint q_base       = qh * args.head_dim;

    /* Cache this Q head in private memory. */
    float q_head[MAX_HEAD_DIM];
    for (uint d = 0; d < args.head_dim; ++d) {
        q_head[d] = q[q_base + d];
    }

    /* Stage 1: each lane finds its local max over positions lane, lane+32, ...
     * then reduce to the global max across the SIMD group. */
    float local_max = -FLT_MAX;
    for (uint pos = lane; pos < seq_len; pos += 32) {
        const uint k_row = pos * kv_stride + kvh * args.head_dim;
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) {
            score += q_head[d] * float(k_cache[k_row + d]);
        }
        score *= scale;
        if (score > local_max) local_max = score;
    }
    const float global_max = simd_max(local_max);

    /* Stage 2: compute exp weights and weighted sum of V, then reduce. */
    float sum_exp = 0.0f;
    float out_head[MAX_HEAD_DIM];
    for (uint d = 0; d < args.head_dim; ++d) out_head[d] = 0.0f;

    for (uint pos = lane; pos < seq_len; pos += 32) {
        const uint kv_row = pos * kv_stride + kvh * args.head_dim;
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) {
            score += q_head[d] * float(k_cache[kv_row + d]);
        }
        score *= scale;
        float w = exp(score - global_max);
        sum_exp += w;
        for (uint d = 0; d < args.head_dim; ++d) {
            out_head[d] += w * float(v_cache[kv_row + d]);
        }
    }

    const float total_exp = simd_sum(sum_exp);
    const float inv_sum   = 1.0f / total_exp;

    /* Only lane 0 writes the final output; simd_sum must still be executed by
     * all lanes, so the value is computed everywhere but the store is guarded. */
    for (uint d = 0; d < args.head_dim; ++d) {
        float val = simd_sum(out_head[d]) * inv_sum;
        if (lane == 0) output[q_base + d] = val;
    }
}

/* -------------------------------------------------------------------------- */
/* 4. Naive causal chunk attention for prefill (O(n²) within the chunk)       */
/* -------------------------------------------------------------------------- */

struct ds3_attention_chunk_args {
    uint32_t seq_pos;      // number of tokens already in the cache before this chunk
    uint32_t n_tokens;     // number of tokens in this chunk
    uint32_t max_seq_len;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
};

// Computes attention for each token in the chunk causally.
//   q        : [n_tokens][n_q_heads][head_dim] FP32 (already RoPE'd)
//   k_cache  : [max_seq_len][n_kv_heads][head_dim] FP16 (past + current chunk)
//   v_cache  : [max_seq_len][n_kv_heads][head_dim] FP16 (past + current chunk)
//   output   : [n_tokens][n_q_heads][head_dim] FP32
//
// Each thread handles one (token, q_head). It attends to positions
// [0 .. seq_pos+token], reading *all* K/V from the FP16 cache. The caller must
// write the current chunk into the cache before dispatching this kernel so that
// the chunk path uses the same FP16 rounding as the token-by-token decode path.
kernel void attention_chunk_gqa(
    device const float        * q         [[buffer(0)]],
    device const half         * k_cache   [[buffer(1)]],
    device const half         * v_cache   [[buffer(2)]],
    device       float        * output    [[buffer(3)]],
    constant ds3_attention_chunk_args & args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint t   = tgpig.x * ntg.x + tpitg.x;
    const uint qh  = tgpig.y * ntg.y + tpitg.y;
    if (t >= args.n_tokens || qh >= args.n_q_heads) return;

    const uint heads_per_kv = args.n_q_heads / args.n_kv_heads;
    const uint kvh          = qh / heads_per_kv;
    const uint kv_stride    = args.n_kv_heads * args.head_dim;
    const uint q_base       = (t * args.n_q_heads + qh) * args.head_dim;
    const uint total_len    = args.seq_pos + t + 1;
    const float scale       = 1.0f / sqrt((float)args.head_dim);

    float q_head[MAX_HEAD_DIM];
    for (uint d = 0; d < args.head_dim; ++d) {
        q_head[d] = q[q_base + d];
    }

    /* First pass: max score for numerical stability. */
    float max_score = -FLT_MAX;
    for (uint pos = 0; pos < total_len; ++pos) {
        const uint k_row = pos * kv_stride + kvh * args.head_dim;
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) {
            score += q_head[d] * float(k_cache[k_row + d]);
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    /* Second pass: softmax denominator and weighted sum of V. */
    float sum_exp = 0.0f;
    float out_head[MAX_HEAD_DIM];
    for (uint d = 0; d < args.head_dim; ++d) out_head[d] = 0.0f;

    for (uint pos = 0; pos < total_len; ++pos) {
        const uint kv_row = pos * kv_stride + kvh * args.head_dim;
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) {
            score += q_head[d] * float(k_cache[kv_row + d]);
        }
        score *= scale;
        float w = exp(score - max_score);
        sum_exp += w;
        for (uint d = 0; d < args.head_dim; ++d) {
            out_head[d] += w * float(v_cache[kv_row + d]);
        }
    }

    const float inv_sum = 1.0f / sum_exp;
    for (uint d = 0; d < args.head_dim; ++d) {
        output[q_base + d] = out_head[d] * inv_sum;
    }
}
