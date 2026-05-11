// MoE FFN kernels for Qwen3.
//
// Each routed expert is a SwiGLU block:
//   gate = silu(input @ W_gate^T)
//   up   =       input @ W_up^T
//   hidden = gate * up
//   out  = hidden @ W_down^T
//   output += weight * out
//
// For decode (single token) we dispatch two kernels per selected expert:
//   1. moe_expert_gate_up_f32 : input -> hidden
//   2. moe_expert_down_f32    : hidden -> output (accumulates)

#include <metal_stdlib>
using namespace metal;

struct ds3_moe_gate_up_args {
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint64_t gate_offset;  // byte offset of this expert in w_gate_exps
    uint64_t up_offset;    // byte offset of this expert in w_up_exps
};

struct ds3_moe_down_args {
    uint32_t n_embd;
    uint32_t n_ff_exp;
    float    weight;       // router weight for this expert
    uint64_t down_offset;  // byte offset of this expert in w_down_exps
};

static float silu(float x) {
    return x / (1.0f + exp(-x));
}

/* Compute hidden = silu(input @ W_gate^T) * (input @ W_up^T).
 * One thread per hidden element. */
kernel void moe_expert_gate_up_f32(
    device const float        * input      [[buffer(0)]],
    device const float        * w_gate     [[buffer(1)]],
    device const float        * w_up       [[buffer(2)]],
    device       float        * hidden     [[buffer(3)]],
    constant ds3_moe_gate_up_args & args   [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint j = tgpig.x * ntg.x + tpitg.x;
    if (j >= args.n_ff_exp) return;

    device const float *gate_row = (device const float *)((device const char *)w_gate + args.gate_offset)
                                   + j * args.n_embd;
    device const float *up_row   = (device const float *)((device const char *)w_up   + args.up_offset)
                                   + j * args.n_embd;

    float gate = 0.0f;
    float up   = 0.0f;
    for (uint d = 0; d < args.n_embd; ++d) {
        gate += input[d] * gate_row[d];
        up   += input[d] * up_row[d];
    }

    hidden[j] = silu(gate) * up;
}

/* Compute output += weight * (hidden @ W_down^T).
 * One thread per output element. */
kernel void moe_expert_down_f32(
    device const float        * hidden     [[buffer(0)]],
    device const float        * w_down     [[buffer(1)]],
    device       float        * output     [[buffer(2)]],
    constant ds3_moe_down_args & args      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint i = tgpig.x * ntg.x + tpitg.x;
    if (i >= args.n_embd) return;

    device const float *down_row = (device const float *)((device const char *)w_down + args.down_offset)
                                   + i * args.n_ff_exp;

    float out = 0.0f;
    for (uint j = 0; j < args.n_ff_exp; ++j) {
        out += hidden[j] * down_row[j];
    }

    output[i] += args.weight * out;
}

/* elementwise: gate = silu(gate) * up. One thread per element. */
struct ds3_moe_silu_mul_args {
    uint32_t n;
};

kernel void moe_expert_silu_mul_f32(
    device const float        * gate [[buffer(0)]],
    device const float        * up   [[buffer(1)]],
    device       float        * out  [[buffer(2)]],
    constant ds3_moe_silu_mul_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint i = tgpig.x * ntg.x + tpitg.x;
    if (i >= args.n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

/* elementwise: output += weight * x. One thread per element. */
struct ds3_moe_accumulate_args {
    uint32_t n;
    float    weight;
};

kernel void moe_expert_accumulate_f32(
    device const float        * x      [[buffer(0)]],
    device       float        * output [[buffer(1)]],
    constant ds3_moe_accumulate_args & args [[buffer(2)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]])
{
    const uint i = tgpig.x * ntg.x + tpitg.x;
    if (i >= args.n) return;
    output[i] += args.weight * x[i];
}

/* ============================================================================
 * Quantized fused MoE expert kernels.
 *
 * These mirror the FP32 fused kernels above but use the SIMD-group reduction
 * style from matmul_q4k/q6k/q8_0. The block structs and block_dot_* helpers are
 * already in scope because ds3_metal concatenates the metal/ sources into one
 * library; they are not duplicated here to avoid redefinition errors.
 * ============================================================================ */

/* Compute hidden = silu(input @ W_gate^T) * (input @ W_up^T).
 * Gate and up weights must share the same quantized type. */
kernel void moe_expert_gate_up_q4k_f32(
    device const float        * input      [[buffer(0)]],
    device const char         * w_gate     [[buffer(1)]],
    device const char         * w_up       [[buffer(2)]],
    device       float        * hidden     [[buffer(3)]],
    constant ds3_moe_gate_up_args & args   [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.n_embd / Q4K_BLOCK_SIZE);
    const int lanes_per_row = q4k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.n_ff_exp) return;

    const uint64_t row_stride = (uint64_t)(args.n_embd / Q4K_BLOCK_SIZE) * sizeof(ds3_block_q4_K);
    device const ds3_block_q4_K *gate_blocks = (device const ds3_block_q4_K *)(w_gate + args.gate_offset + row * row_stride);
    device const ds3_block_q4_K *up_blocks   = (device const ds3_block_q4_K *)(w_up   + args.up_offset   + row * row_stride);

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        gate_partial += block_dot_q4k(&gate_blocks[b], &input[b * Q4K_BLOCK_SIZE]);
        up_partial   += block_dot_q4k(&up_blocks[b],   &input[b * Q4K_BLOCK_SIZE]);
    }

    float gate_sum = gate_partial;
    float up_sum   = up_partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            gate_sum += simd_shuffle_down(gate_sum, offset);
            up_sum   += simd_shuffle_down(up_sum,   offset);
        }
    }

    if (block_in_row == 0) {
        hidden[row] = silu(gate_sum) * up_sum;
    }
}

kernel void moe_expert_gate_up_q6k_f32(
    device const float        * input      [[buffer(0)]],
    device const char         * w_gate     [[buffer(1)]],
    device const char         * w_up       [[buffer(2)]],
    device       float        * hidden     [[buffer(3)]],
    constant ds3_moe_gate_up_args & args   [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.n_embd / Q6K_BLOCK_SIZE);
    const int lanes_per_row = q6k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.n_ff_exp) return;

    const uint64_t row_stride = (uint64_t)(args.n_embd / Q6K_BLOCK_SIZE) * sizeof(ds3_block_q6_K);
    device const ds3_block_q6_K *gate_blocks = (device const ds3_block_q6_K *)(w_gate + args.gate_offset + row * row_stride);
    device const ds3_block_q6_K *up_blocks   = (device const ds3_block_q6_K *)(w_up   + args.up_offset   + row * row_stride);

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        gate_partial += block_dot_q6k(&gate_blocks[b], &input[b * Q6K_BLOCK_SIZE]);
        up_partial   += block_dot_q6k(&up_blocks[b],   &input[b * Q6K_BLOCK_SIZE]);
    }

    float gate_sum = gate_partial;
    float up_sum   = up_partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            gate_sum += simd_shuffle_down(gate_sum, offset);
            up_sum   += simd_shuffle_down(up_sum,   offset);
        }
    }

    if (block_in_row == 0) {
        hidden[row] = silu(gate_sum) * up_sum;
    }
}

kernel void moe_expert_gate_up_q8_0_f32(
    device const float        * input      [[buffer(0)]],
    device const char         * w_gate     [[buffer(1)]],
    device const char         * w_up       [[buffer(2)]],
    device       float        * hidden     [[buffer(3)]],
    constant ds3_moe_gate_up_args & args   [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.n_embd / Q8_0_BLOCK_SIZE);
    const int lanes_per_row = q8_0_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.n_ff_exp) return;

    const uint64_t row_stride = (uint64_t)(args.n_embd / Q8_0_BLOCK_SIZE) * sizeof(ds3_block_q8_0);
    device const ds3_block_q8_0 *gate_blocks = (device const ds3_block_q8_0 *)(w_gate + args.gate_offset + row * row_stride);
    device const ds3_block_q8_0 *up_blocks   = (device const ds3_block_q8_0 *)(w_up   + args.up_offset   + row * row_stride);

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        gate_partial += block_dot_q8_0(&gate_blocks[b], &input[b * Q8_0_BLOCK_SIZE]);
        up_partial   += block_dot_q8_0(&up_blocks[b],   &input[b * Q8_0_BLOCK_SIZE]);
    }

    float gate_sum = gate_partial;
    float up_sum   = up_partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            gate_sum += simd_shuffle_down(gate_sum, offset);
            up_sum   += simd_shuffle_down(up_sum,   offset);
        }
    }

    if (block_in_row == 0) {
        hidden[row] = silu(gate_sum) * up_sum;
    }
}

/* Compute output += weight * (hidden @ W_down^T). */
kernel void moe_expert_down_q4k_f32(
    device const float        * hidden     [[buffer(0)]],
    device const char         * w_down     [[buffer(1)]],
    device       float        * output     [[buffer(2)]],
    constant ds3_moe_down_args & args      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.n_ff_exp / Q4K_BLOCK_SIZE);
    const int lanes_per_row = q4k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.n_embd) return;

    const uint64_t row_stride = (uint64_t)(args.n_ff_exp / Q4K_BLOCK_SIZE) * sizeof(ds3_block_q4_K);
    device const ds3_block_q4_K *down_blocks = (device const ds3_block_q4_K *)(w_down + args.down_offset + row * row_stride);

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q4k(&down_blocks[b], &hidden[b * Q4K_BLOCK_SIZE]);
    }

    float sum = partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            sum += simd_shuffle_down(sum, offset);
        }
    }

    if (block_in_row == 0) {
        output[row] += args.weight * sum;
    }
}

kernel void moe_expert_down_q6k_f32(
    device const float        * hidden     [[buffer(0)]],
    device const char         * w_down     [[buffer(1)]],
    device       float        * output     [[buffer(2)]],
    constant ds3_moe_down_args & args      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.n_ff_exp / Q6K_BLOCK_SIZE);
    const int lanes_per_row = q6k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.n_embd) return;

    const uint64_t row_stride = (uint64_t)(args.n_ff_exp / Q6K_BLOCK_SIZE) * sizeof(ds3_block_q6_K);
    device const ds3_block_q6_K *down_blocks = (device const ds3_block_q6_K *)(w_down + args.down_offset + row * row_stride);

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q6k(&down_blocks[b], &hidden[b * Q6K_BLOCK_SIZE]);
    }

    float sum = partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            sum += simd_shuffle_down(sum, offset);
        }
    }

    if (block_in_row == 0) {
        output[row] += args.weight * sum;
    }
}

kernel void moe_expert_down_q8_0_f32(
    device const float        * hidden     [[buffer(0)]],
    device const char         * w_down     [[buffer(1)]],
    device       float        * output     [[buffer(2)]],
    constant ds3_moe_down_args & args      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const int n_blocks      = int(args.n_ff_exp / Q8_0_BLOCK_SIZE);
    const int lanes_per_row = q8_0_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tiisg);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= args.n_embd) return;

    const uint64_t row_stride = (uint64_t)(args.n_ff_exp / Q8_0_BLOCK_SIZE) * sizeof(ds3_block_q8_0);
    device const ds3_block_q8_0 *down_blocks = (device const ds3_block_q8_0 *)(w_down + args.down_offset + row * row_stride);

    float partial = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        partial += block_dot_q8_0(&down_blocks[b], &hidden[b * Q8_0_BLOCK_SIZE]);
    }

    float sum = partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            sum += simd_shuffle_down(sum, offset);
        }
    }

    if (block_in_row == 0) {
        output[row] += args.weight * sum;
    }
}

/* ============================================================================
 * GPU-only routed MoE path for Qwen3-30B-A3B (two-pass, no CPU readback).
 *
 * Eliminates the per-layer CPU router readback by keeping the router softmax,
 * top-k and the 8 selected experts entirely on the GPU.
 *
 *   Pass 1: router matmul + softmax + top-k + gate/up projection -> hidden
 *   Pass 2: down projection: output += weight * (hidden @ W_down^T)
 *
 * The intermediate hidden state is written to a small device buffer, which lets
 * both passes use full GPU parallelism instead of fitting everything into one
 * oversized threadgroup.
 *
 * Gate/up are Q4_K and down is Q6_K, matching the real Qwen3 model. The caller
 * is responsible for zeroing `output`; the kernel accumulates routed expert
 * contributions so that a separately-dispatched shared expert can be added.
 * ============================================================================ */

struct ds3_moe_routed_args {
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t norm_topk_prob;
};

#define MOE_MAX_N_USED 8
#define MOE_MAX_N_FF   768

static inline void moe_router_topk(
    device const float        * input,
    device const float        * w_gate_inp,
    threadgroup float         * sh_logits,
    threadgroup int32_t       * sh_indices,
    threadgroup float         * sh_scores,
    uint n_embd,
    uint n_expert,
    uint n_used,
    bool norm_topk_prob,
    uint tid,
    uint n_threads)
{
    /* Router logits. */
    for (uint e = tid; e < n_expert; e += n_threads) {
        float sum = 0.0f;
        device const float *row = w_gate_inp + e * n_embd;
        for (uint d = 0; d < n_embd; ++d) {
            sum += input[d] * row[d];
        }
        sh_logits[e] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Softmax + top-k on a single thread (n_expert/n_used are tiny). */
    if (tid == 0) {
        float max_logit = -INFINITY;
        for (uint e = 0; e < n_expert; ++e) {
            max_logit = max(max_logit, sh_logits[e]);
        }

        float sum = 0.0f;
        for (uint e = 0; e < n_expert; ++e) {
            sh_logits[e] = exp(sh_logits[e] - max_logit);
            sum += sh_logits[e];
        }
        float inv_sum = 1.0f / sum;
        for (uint e = 0; e < n_expert; ++e) {
            sh_logits[e] *= inv_sum;
        }

        for (uint k = 0; k < n_used; ++k) {
            float best_p = -1.0f;
            uint  best_e = 0;
            for (uint e = 0; e < n_expert; ++e) {
                if (sh_logits[e] > best_p) {
                    best_p = sh_logits[e];
                    best_e = e;
                }
            }
            sh_indices[k] = (int32_t)best_e;
            sh_scores[k]  = best_p;
            sh_logits[best_e] = -1.0f;
        }

        if (norm_topk_prob) {
            float s = 0.0f;
            for (uint k = 0; k < n_used; ++k) s += sh_scores[k];
            if (s > 0.0f) {
                float inv = 1.0f / s;
                for (uint k = 0; k < n_used; ++k) sh_scores[k] *= inv;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* Batch router softmax + top-k for many layers.
 * Input:  logits [n_layers][n_expert]
 * Output: indices[n_layers][n_used], scores[n_layers][n_used]
 * One threadgroup per layer. */
struct ds3_moe_router_batch_args {
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t norm_topk_prob;
};

kernel void moe_router_topk_batch_f32(
    device const float        * logits  [[buffer(0)]],
    device       int32_t      * indices [[buffer(1)]],
    device       float        * scores  [[buffer(2)]],
    constant ds3_moe_router_batch_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint n_expert = args.n_expert;
    const uint n_used   = args.n_used;
    const uint layer    = tgpig.x;

    device const float *layer_logits = logits + layer * n_expert;
    device       int32_t *layer_indices = indices + layer * n_used;
    device       float   *layer_scores  = scores  + layer * n_used;

    float max_logit = -INFINITY;
    for (uint i = 0; i < n_expert; i++) {
        max_logit = max(max_logit, layer_logits[i]);
    }

    float sum = 0.0f;
    float probs[128];
    for (uint i = 0; i < n_expert; i++) {
        probs[i] = exp(layer_logits[i] - max_logit);
        sum += probs[i];
    }

    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < n_expert; i++) {
        probs[i] *= inv_sum;
    }

    for (uint k = 0; k < n_used; k++) {
        float best_p = -1.0f;
        uint  best_i = 0;
        for (uint i = 0; i < n_expert; i++) {
            if (probs[i] > best_p) {
                best_p = probs[i];
                best_i = i;
            }
        }
        layer_indices[k] = (int32_t)best_i;
        layer_scores[k]  = best_p;
        probs[best_i] = -1.0f;
    }

    if (args.norm_topk_prob) {
        float s = 0.0f;
        for (uint k = 0; k < n_used; k++) s += layer_scores[k];
        if (s > 0.0f) {
            float inv = 1.0f / s;
            for (uint k = 0; k < n_used; k++) layer_scores[k] *= inv;
        }
    }
}

/* Batch router softmax + top-k for many layers and many tokens.
 * Input:  logits [n_layers][n_tokens][n_expert]
 * Output: indices[n_layers][n_tokens][n_used], scores[n_layers][n_tokens][n_used]
 * One threadgroup per (layer, token) pair. */
struct ds3_moe_router_batch_tokens_args {
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t n_tokens;
    uint32_t norm_topk_prob;
};

kernel void moe_router_topk_batch_tokens_f32(
    device const float        * logits  [[buffer(0)]],
    device       int32_t      * indices [[buffer(1)]],
    device       float        * scores  [[buffer(2)]],
    constant ds3_moe_router_batch_tokens_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint n_expert = args.n_expert;
    const uint n_used   = args.n_used;
    const uint n_tokens = args.n_tokens;
    const uint layer    = tgpig.x;
    const uint token    = tgpig.y;
    if (token >= n_tokens) return;

    const uint row_stride = n_tokens * n_expert;
    device const float *row_logits = logits + layer * row_stride + token * n_expert;
    device       int32_t *row_indices = indices + layer * row_stride + token * n_used;
    device       float   *row_scores  = scores  + layer * row_stride + token * n_used;

    float max_logit = -INFINITY;
    for (uint i = 0; i < n_expert; i++) {
        max_logit = max(max_logit, row_logits[i]);
    }

    float sum = 0.0f;
    float probs[128];
    for (uint i = 0; i < n_expert; i++) {
        probs[i] = exp(row_logits[i] - max_logit);
        sum += probs[i];
    }

    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < n_expert; i++) {
        probs[i] *= inv_sum;
    }

    for (uint k = 0; k < n_used; k++) {
        float best_p = -1.0f;
        uint  best_i = 0;
        for (uint i = 0; i < n_expert; i++) {
            if (probs[i] > best_p) {
                best_p = probs[i];
                best_i = i;
            }
        }
        row_indices[k] = (int32_t)best_i;
        row_scores[k]  = best_p;
        probs[best_i] = -1.0f;
    }

    if (args.norm_topk_prob) {
        float s = 0.0f;
        for (uint k = 0; k < n_used; k++) s += row_scores[k];
        if (s > 0.0f) {
            float inv = 1.0f / s;
            for (uint k = 0; k < n_used; k++) row_scores[k] *= inv;
        }
    }
}

/* Pass 1: router + hidden. One SIMD group per hidden element. */
kernel void moe_router_hidden_q4k_f32(
    device const float        * input      [[buffer(0)]],
    device const float        * w_gate_inp [[buffer(1)]],
    device const char         * w_gate     [[buffer(2)]],
    device const char         * w_up       [[buffer(3)]],
    device const uint64_t     * offsets    [[buffer(4)]],  // 3 * n_expert
    device       float        * hidden     [[buffer(5)]],  // [n_used][n_ff]
    constant ds3_moe_routed_args & args    [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const uint n_embd   = args.n_embd;
    const uint n_ff     = args.n_ff_exp;
    const uint n_expert = args.n_expert;
    const uint n_used   = args.n_used;

    const uint tid = tiisg;
    const uint n_threads = 32;

    threadgroup float   sh_logits[128];
    threadgroup int32_t sh_indices[MOE_MAX_N_USED];
    threadgroup float   sh_scores [MOE_MAX_N_USED];

    moe_router_topk(input, w_gate_inp,
                    sh_logits, sh_indices, sh_scores,
                    n_embd, n_expert, n_used,
                    args.norm_topk_prob != 0,
                    tid, n_threads);

    const int n_blocks      = int(n_embd / Q4K_BLOCK_SIZE);
    const int lanes_per_row = q4k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tid);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int hid = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (hid >= int(n_used * n_ff)) return;

    const uint k = (uint)hid / n_ff;
    const uint j = (uint)hid % n_ff;
    const uint expert = (uint)sh_indices[k];

    const uint64_t gate_row_stride = (uint64_t)(n_embd / Q4K_BLOCK_SIZE) * sizeof(ds3_block_q4_K);
    const uint64_t gate_off = offsets[expert * 3 + 0];
    const uint64_t up_off   = offsets[expert * 3 + 1];

    device const ds3_block_q4_K *gate_blocks = (device const ds3_block_q4_K *)(w_gate + gate_off + j * gate_row_stride);
    device const ds3_block_q4_K *up_blocks   = (device const ds3_block_q4_K *)(w_up   + up_off   + j * gate_row_stride);

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;
    for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
        gate_partial += block_dot_q4k(&gate_blocks[b], &input[b * Q4K_BLOCK_SIZE]);
        up_partial   += block_dot_q4k(&up_blocks[b],   &input[b * Q4K_BLOCK_SIZE]);
    }

    float gate_sum = gate_partial;
    float up_sum   = up_partial;
    for (int offset = 1; offset < lanes_per_row; offset *= 2) {
        if (block_in_row + offset < lanes_per_row) {
            gate_sum += simd_shuffle_down(gate_sum, offset);
            up_sum   += simd_shuffle_down(up_sum,   offset);
        }
    }

    if (block_in_row == 0) {
        hidden[hid] = silu(gate_sum) * up_sum;
    }
}

/* Pass 2: down projection. One SIMD group per output element. */
kernel void moe_output_q6k_f32(
    device const float        * input      [[buffer(0)]],
    device const float        * w_gate_inp [[buffer(1)]],
    device const float        * hidden     [[buffer(2)]],  // [n_used][n_ff]
    device const char         * w_down     [[buffer(3)]],
    device const uint64_t     * offsets    [[buffer(4)]],  // 3 * n_expert
    device       float        * output     [[buffer(5)]],
    constant ds3_moe_routed_args & args    [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const uint n_embd   = args.n_embd;
    const uint n_ff     = args.n_ff_exp;
    const uint n_expert = args.n_expert;
    const uint n_used   = args.n_used;

    const uint tid = tiisg;
    const uint n_threads = 32;

    threadgroup float   sh_logits[128];
    threadgroup int32_t sh_indices[MOE_MAX_N_USED];
    threadgroup float   sh_scores [MOE_MAX_N_USED];

    moe_router_topk(input, w_gate_inp,
                    sh_logits, sh_indices, sh_scores,
                    n_embd, n_expert, n_used,
                    args.norm_topk_prob != 0,
                    tid, n_threads);

    const int n_blocks      = int(n_ff / Q6K_BLOCK_SIZE);
    const int lanes_per_row = q6k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tid);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= int(n_embd)) return;

    const uint64_t down_row_stride = (uint64_t)(n_ff / Q6K_BLOCK_SIZE) * sizeof(ds3_block_q6_K);

    float out = 0.0f;
    for (uint k = 0; k < n_used; ++k) {
        const uint expert = (uint)sh_indices[k];
        const uint64_t down_off = offsets[expert * 3 + 2];
        device const ds3_block_q6_K *down_blocks = (device const ds3_block_q6_K *)(w_down + down_off + row * down_row_stride);

        float partial = 0.0f;
        for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
            partial += block_dot_q6k(&down_blocks[b], &hidden[k * n_ff + b * Q6K_BLOCK_SIZE]);
        }

        float sum = partial;
        for (int offset = 1; offset < lanes_per_row; offset *= 2) {
            if (block_in_row + offset < lanes_per_row) {
                sum += simd_shuffle_down(sum, offset);
            }
        }

        if (block_in_row == 0) {
            out += sh_scores[k] * sum;
        }
    }

    if (block_in_row == 0) {
        output[row] += out;
    }
}

/* Pass 2: down projection for Q4_K expert down weights. */
kernel void moe_output_q4k_f32(
    device const float        * input      [[buffer(0)]],
    device const float        * w_gate_inp [[buffer(1)]],
    device const float        * hidden     [[buffer(2)]],  // [n_used][n_ff]
    device const char         * w_down     [[buffer(3)]],
    device const uint64_t     * offsets    [[buffer(4)]],  // 3 * n_expert
    device       float        * output     [[buffer(5)]],
    constant ds3_moe_routed_args & args    [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    const uint n_embd   = args.n_embd;
    const uint n_ff     = args.n_ff_exp;
    const uint n_expert = args.n_expert;
    const uint n_used   = args.n_used;

    const uint tid = tiisg;
    const uint n_threads = 32;

    threadgroup float   sh_logits[128];
    threadgroup int32_t sh_indices[MOE_MAX_N_USED];
    threadgroup float   sh_scores [MOE_MAX_N_USED];

    moe_router_topk(input, w_gate_inp,
                    sh_logits, sh_indices, sh_scores,
                    n_embd, n_expert, n_used,
                    args.norm_topk_prob != 0,
                    tid, n_threads);

    const int n_blocks      = int(n_ff / Q4K_BLOCK_SIZE);
    const int lanes_per_row = q4k_lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    const int lane         = int(tid);
    const int row_in_simd  = lane / lanes_per_row;
    const int block_in_row = lane % lanes_per_row;

    const int row = int(tgpig.x) * rows_per_simd + row_in_simd;
    if (row >= int(n_embd)) return;

    const uint64_t down_row_stride = (uint64_t)(n_ff / Q4K_BLOCK_SIZE) * sizeof(ds3_block_q4_K);

    float out = 0.0f;
    for (uint k = 0; k < n_used; ++k) {
        const uint expert = (uint)sh_indices[k];
        const uint64_t down_off = offsets[expert * 3 + 2];
        device const ds3_block_q4_K *down_blocks = (device const ds3_block_q4_K *)(w_down + down_off + row * down_row_stride);

        float partial = 0.0f;
        for (int b = block_in_row; b < n_blocks; b += lanes_per_row) {
            partial += block_dot_q4k(&down_blocks[b], &hidden[k * n_ff + b * Q4K_BLOCK_SIZE]);
        }

        float sum = partial;
        for (int offset = 1; offset < lanes_per_row; offset *= 2) {
            if (block_in_row + offset < lanes_per_row) {
                sum += simd_shuffle_down(sum, offset);
            }
        }

        if (block_in_row == 0) {
            out += sh_scores[k] * sum;
        }
    }

    if (block_in_row == 0) {
        output[row] += out;
    }
}
