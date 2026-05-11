/*
 * ds3_reference.h — CPU FP32 Reference Implementation for Qwen3 Layer Forward
 *
 * Provides baseline correctness for Metal kernels.
 * All operations are single-threaded FP32 for clarity.
 *
 * Usage flow (single token, Layer 0):
 *   1. ds3_ref_dequantize_tensor()  — load weights from GGUF → FP32 buffers
 *   2. ds3_ref_layer_forward()      — run one layer
 *   3. Compare against Python/llama.cpp ground truth
 */

#ifndef DS3_REFERENCE_H
#define DS3_REFERENCE_H

#include "ds3.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Dequantization
 * ============================================================================ */

/* Dequantize a single GGUF tensor into a freshly-allocated FP32 buffer.
 * Returns NULL on unsupported type or allocation failure.
 * Caller must free() the returned pointer. */
float *ds3_ref_dequantize_tensor(const ds3_tensor_t *t);

/* Dequantize one row of a GGUF tensor into caller-supplied FP32 buffer.
 *   t   : tensor with at least 2 dimensions [rows][cols]
 *   row : row index (0 <= row < t->ne[1])
 *   out : caller-allocated [cols] floats
 * Returns 0 on success, -1 on error.
 *
 * Used for token embedding lookup when token_embd is quantized. */
int ds3_ref_dequantize_row(const ds3_tensor_t *t, int row, float *out);

/* ============================================================================
 * 2. Basic Operators
 * ============================================================================ */

/* RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps) */
void ds3_ref_rms_norm(const float *x, const float *weight, float eps,
                      int n, float *out);

/* In-place softmax over [0..n-1] */
void ds3_ref_softmax(float *x, int n);

/* RoPE — standard full-dimension, theta = 1e6.
 *
 * q: [n_q_head][head_dim]   — interleaved (head-major)
 * k: [n_kv_head][head_dim]  — interleaved (head-major)
 *
 * seq_pos: absolute position of the current token.
 *
 * Applies in-place. */
void ds3_ref_rope(float *q, float *k,
                  int seq_pos,
                  int n_q_head, int n_kv_head, int head_dim);

/* General matrix multiply: C = A × B
 * A: [M][K], B: [K][N], C: [M][N]
 * All row-major. */
void ds3_ref_matmul(const float *A, const float *B, float *C,
                    int M, int N, int K);

/* Vector×matrix: y = x @ W^T
 * x: [in_dim], W: [out_dim][in_dim] row-major, y: [out_dim]
 * Used for all linear projections. */
void ds3_ref_vec_matmul(const float *x, const float *W, float *y,
                        int in_dim, int out_dim);

/* SiLU: out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])) */
void ds3_ref_silu(const float *x, float *out, int n);

/* Element-wise multiply: out[i] = a[i] * b[i] */
void ds3_ref_mul(const float *a, const float *b, float *out, int n);

/* Copy: out[i] = in[i] */
void ds3_ref_copy(const float *in_, float *out, int n);

/* Add: out[i] = a[i] + b[i] */
void ds3_ref_add(const float *a, const float *b, float *out, int n);

/* Top-k (for MoE routing).
 * scores: [N_EXPERT]
 * k: number of elements to select
 * indices_out: [k]  — indices of top-k elements (sorted descending)
 * values_out:  [k]  — corresponding scores */
void ds3_ref_topk(const float *scores, int n, int k,
                  int *indices_out, float *values_out);

/* ============================================================================
 * 3. Attention
 * ============================================================================ */

/* Single-token GQA attention with KV cache.
 *
 * Inputs (FP32, already dequantized):
 *   input:     [N_EMBD]           — post-norm hidden state
 *   w_q:       [Q_DIM][N_EMBD]    — Q projection weight
 *   w_q_norm:  [HEAD_DIM]         — Q RMSNorm weight (if HAS_QK_NORM)
 *   w_k:       [KV_DIM][N_EMBD]   — K projection weight
 *   w_k_norm:  [HEAD_DIM]         — K RMSNorm weight (if HAS_QK_NORM)
 *   w_v:       [KV_DIM][N_EMBD]   — V projection weight
 *   w_o:       [N_EMBD][Q_DIM]    — O projection weight
 *
 *   kv_cache:  current layer's KV cache (k_cache + v_cache)
 *   seq_pos:   current token position (0 = first token)
 *   n_tokens:  number of tokens in current prefill chunk (1 for decode)
 *
 * Output:
 *   output:    [N_EMBD]
 *
 * scratch: caller-allocated temporaries:
 *   q: [Q_DIM]  (aligned for head splitting)
 *   k: [KV_DIM]
 *   v: [KV_DIM]
 *   attn_scores: [DS3_N_HEAD][seq_len + n_tokens]
 */
void ds3_ref_gqa_attention(
    const float *input,
    const float *w_q, const float *w_q_norm,
    const float *w_k, const float *w_k_norm,
    const float *w_v,
    const float *w_o,
    ds3_kv_cache_t *kv_cache,
    int seq_pos, int n_tokens,
    float *output,
    float *q, float *k, float *v,
    float *attn_scores);

/* ============================================================================
 * 4. MoE FFN
 * ============================================================================ */

/* MoE FFN forward.
 *
 * Inputs:
 *   input:          [N_EMBD]   — post-attention hidden state (post-norm)
 *   w_gate_inp:     [N_EXPERT][N_EMBD]      — router weight
 *   w_gate_exps:    [N_EXPERT][N_FF_EXP][N_EMBD] — expert gate weights
 *   w_up_exps:      [N_EXPERT][N_FF_EXP][N_EMBD] — expert up weights
 *   w_down_exps:    [N_EXPERT][N_EMBD][N_FF_EXP] — expert down weights
 *   w_shared_gate:  [N_FF_SHARED][N_EMBD]   — shared expert gate (optional)
 *   w_shared_up:    [N_FF_SHARED][N_EMBD]   — shared expert up (optional)
 *   w_shared_down:  [N_EMBD][N_FF_SHARED]   — shared expert down (optional)
 *
 * Output:
 *   output: [N_EMBD]
 *
 * scratch: caller-allocated temporaries
 *   gate_logits:   [N_EXPERT]
 *   expert_scores: [N_EXPERT_USED]
 *   expert_indices:[N_EXPERT_USED]
 *
 * NOTE: SwiGLU temporaries are allocated internally via calloc.
 */
void ds3_ref_moe_ffn(
    const float *input,
    const float *w_gate_inp,
    const float *w_gate_exps,
    const float *w_up_exps,
    const float *w_down_exps,
    const float *w_shared_gate,
    const float *w_shared_up,
    const float *w_shared_down,
    float *output,
    float *gate_logits,
    float *expert_scores,
    int *expert_indices,
    int n_embd,
    int n_expert,
    int n_used,
    int n_ff_exp,
    int n_ff_shared,
    bool norm_topk_prob);

/* ============================================================================
 * 5. Single-Layer Forward
 * ============================================================================ */

/* Layer 0 forward — the integration point.
 *
 * All weights must be pre-dequantized to FP32.
 * KV cache is updated in-place.
 *
 * Flow:
 *   1. residual = input
 *   2. xb = RMSNorm(input)
 *   3. attn_out = GQA_Attention(xb)
 *   4. attn_out += residual
 *   5. residual = attn_out
 *   6. xb = RMSNorm(attn_out)
 *   7. ffn_out = MoE_FFN(xb)
 *   8. output = ffn_out + residual
 */
void ds3_ref_layer_forward(
    const ds3_layer_weights_t *layer,
    const float *input,
    ds3_kv_cache_t *kv_cache,
    int seq_pos, int n_tokens,
    float *output,
    ds3_layer_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* DS3_REFERENCE_H */
