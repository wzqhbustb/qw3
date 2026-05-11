#ifndef DS3_METAL_H
#define DS3_METAL_H

#include <stdint.h>
#include <stddef.h>

#include "ds3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a GPU buffer. */
typedef struct ds3_metal_buffer ds3_metal_buffer_t;

/* Initialize / shutdown the Metal backend.
 * Returns 0 on success, negative on error.
 */
int  ds3_metal_init(void);
void ds3_metal_shutdown(void);

/* Allocate a GPU buffer with `bytes` bytes.
 * Returns NULL on failure.
 */
ds3_metal_buffer_t * ds3_metal_buffer_alloc(size_t bytes);
void                 ds3_metal_buffer_free(ds3_metal_buffer_t *buf);

/* Create a read-only GPU buffer that aliases an existing mmap'd region.
 * `base` must be page-aligned. The mmap lifetime is managed by the caller.
 */
ds3_metal_buffer_t * ds3_metal_buffer_from_mmap(const void *base, size_t bytes);

/* Create a sub-range view of an existing GPU buffer. The view shares the
 * underlying MTLBuffer and applies `offset` at kernel bind time. */
ds3_metal_buffer_t * ds3_metal_buffer_view(const ds3_metal_buffer_t *base,
                                           size_t offset, size_t bytes);

/* Copy data between host and GPU. Writes to mmap-backed buffers are refused. */
int ds3_metal_buffer_write(ds3_metal_buffer_t *buf, size_t offset,
                           const void *src, size_t bytes);
int ds3_metal_buffer_read (ds3_metal_buffer_t *buf, size_t offset,
                           void *dst, size_t bytes);

/* ------------------------------------------------------------------ */
/* Kernel dispatch                                                    */
/* ------------------------------------------------------------------ */

/* RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
 *   x      : [n_rows, n]  f32
 *   weight : [n]          f32
 *   out    : [n_rows, n]  f32
 */
int ds3_metal_rms_norm(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *weight,
    ds3_metal_buffer_t       *out,
    uint32_t n, uint32_t n_rows, float eps);

/* RoPE (full-dim): applies rotary embedding to all head_dim pairs.
 *   src       : [n_rows, n_heads * head_dim]  f32
 *   dst       : [n_rows, n_heads * head_dim]  f32
 *   positions : [n_rows]                      int32  (token positions)
 */
int ds3_metal_rope(
    const ds3_metal_buffer_t *src,
    ds3_metal_buffer_t       *dst,
    const ds3_metal_buffer_t *positions,
    uint32_t n_heads, uint32_t head_dim, uint32_t n_rows,
    float theta_base);

/* Matrix multiply: C = A × B
 *   A: [M][K] row-major
 *   B: [K][N] row-major
 *   C: [M][N] row-major
 */
int ds3_metal_matmul(
    const ds3_metal_buffer_t *A,
    const ds3_metal_buffer_t *B,
    ds3_metal_buffer_t       *C,
    uint32_t M, uint32_t N, uint32_t K);

/* Vector × matrix: y = x @ W^T
 *   x: [in_dim]      FP32
 *   W: [out_dim][in_dim] row-major FP32
 *   y: [out_dim]     FP32
 *   row_stride: byte stride between consecutive rows of W
 *               (must be >= in_dim * sizeof(float);
 *                use in_dim * sizeof(float) for a dense matrix)
 */
int ds3_metal_vec_matmul_f32(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride);

/* Vector × quantized matrix: y = x @ W^T
 *   x: [in_dim]      FP32
 *   W: [out_dim][in_dim] row-major quantized weights (Q4_K or Q8_0)
 *   y: [out_dim]     FP32
 *   row_stride: byte stride between rows of W (e.g. for Q4_K: (in_dim/256)*144)
 */
int ds3_metal_vec_matmul_quantized(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride,
    ds3_type_t type);

/* SIMD-group parallel reduction variants.
 * Same shapes as above, but lanes within a SIMD group split the dot product
 * across columns/blocks and reduce via simd_shuffle_down. Typically faster for
 * large in_dim, but requires the dispatch width to be a multiple of the SIMD
 * group size (32).
 */
int ds3_metal_vec_matmul_f32_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride);

int ds3_metal_vec_matmul_q8_0_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride);

int ds3_metal_vec_matmul_q4k_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride);

int ds3_metal_vec_matmul_q6k_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride);

/* Q4_K with half-precision input activations (e.g. FP16 KV cache).
 * x must be an MTLBuffer of half (FP16) values; W and y are unchanged.
 */
int ds3_metal_vec_matmul_q4k_half(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride);

/* Batched matrix × quantized matrix: C = A @ W^T
 *   A: [M][K] FP32, row-major
 *   W: [N][K] quantized (Q4_K / Q6_K / Q8_0), row-major
 *   C: [M][N] FP32, row-major
 * weight_row_stride: byte stride between rows of W.
 * weight_offset    : byte offset to the start of the matrix inside W. */
int ds3_metal_matmul_quantized_batch(
    const ds3_metal_buffer_t *A,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *C,
    uint32_t M, uint32_t N, uint32_t K,
    uint64_t weight_row_stride,
    uint64_t weight_offset,
    ds3_type_t type);

/* Elementwise FP32 helpers.
 * These are small 1-D kernels that avoid host<->device transfers for residual
 * adds, buffer copies, and zeroing inside the engine graph.
 * `bytes` must be a multiple of sizeof(float).
 */
int ds3_metal_buffer_copy_f32(const ds3_metal_buffer_t *src,
                              ds3_metal_buffer_t       *dst,
                              size_t bytes);
int ds3_metal_buffer_zero_f32(ds3_metal_buffer_t *buf, size_t bytes);
int ds3_metal_vec_add_inplace(ds3_metal_buffer_t       *acc,
                              const ds3_metal_buffer_t *addend,
                              uint32_t n);
int ds3_metal_vec_add(const ds3_metal_buffer_t *a,
                      const ds3_metal_buffer_t *b,
                      ds3_metal_buffer_t       *c,
                      uint32_t n);

/* Gather / scatter rows for batched MoE.
 *   gather: dst[i][j] = src[ ids[offset + i] ][j]
 *   scatter: dst[ ids[offset + i] ][j] += scores[offset + i] * src[i][j]
 */
int ds3_metal_gather_rows_f32(const ds3_metal_buffer_t *src,
                              const ds3_metal_buffer_t *ids,
                              ds3_metal_buffer_t       *dst,
                              uint32_t n_cols, uint32_t count, uint32_t offset);
int ds3_metal_scatter_add_weighted_f32(const ds3_metal_buffer_t *src,
                                       const ds3_metal_buffer_t *scores,
                                       const ds3_metal_buffer_t *ids,
                                       ds3_metal_buffer_t       *dst,
                                       uint32_t n_cols, uint32_t count, uint32_t offset);

/* elementwise: out[i] = silu(gate[i]) * up[i] */
int ds3_metal_silu_mul_f32(const ds3_metal_buffer_t *gate,
                           const ds3_metal_buffer_t *up,
                           ds3_metal_buffer_t       *out,
                           uint32_t n);

/* Decode-only GQA attention.
 *   q, k, v : FP32, already projected, normed, and RoPE'd.
 *   k_cache / v_cache : FP16 KV cache, shape [max_seq_len][n_kv_heads][head_dim].
 *   output  : FP32 [n_q_heads][head_dim].
 * The kernel writes k/v into cache[seq_pos], then computes softmax(Q @ K^T) @ V.
 */
int ds3_metal_attention_decode(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim);

/* SIMD-group parallel variant of the above.
 * One SIMD group (32 lanes) processes one Q head. Typically much higher GPU
 * occupancy than the baseline version for large seq_len.
 */
int ds3_metal_attention_decode_simd(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim);

/* Decode-only GQA attention with integrated RoPE.
 * q and k are projected but NOT yet RoPE'd; v is projected and does not need RoPE.
 * Internally this function applies RoPE to q and k (in-place), writes the new K/V
 * row into the cache, then runs the attention compute kernel.
 *
 * NOTE: q and k are modified in-place by RoPE. If the caller needs the original
 * projected (but unrotated) values, it must make a copy before calling this.
 *
 * All buffers have the same shapes as ds3_metal_attention_decode().
 * theta_base is the RoPE base (e.g. 1e6 for Qwen3).
 */
int ds3_metal_attention_decode_rope(
    ds3_metal_buffer_t       *q,
    ds3_metal_buffer_t       *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float theta_base);

/* SIMD-group parallel variant of attention with integrated RoPE. */
int ds3_metal_attention_decode_rope_simd(
    ds3_metal_buffer_t       *q,
    ds3_metal_buffer_t       *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float theta_base);

/* Batched KV cache write for chunked prefill.
 *   k, v : [n_tokens][n_kv_heads][head_dim] FP32
 *   k_cache / v_cache : FP16 [max_seq_len][n_kv_heads][head_dim]
 * Writes each token at cache position seq_pos + token_index. */
int ds3_metal_kv_cache_write_batch(
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    uint32_t seq_pos,
    uint32_t n_tokens,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim);

/* Naive causal chunk attention for prefill.
 *   q        : [n_tokens][n_q_heads][head_dim] FP32, already RoPE'd
 *   k_chunk  : [n_tokens][n_kv_heads][head_dim] FP32, already RoPE'd
 *   v_chunk  : [n_tokens][n_kv_heads][head_dim] FP32
 *   k_cache  : [max_seq_len][n_kv_heads][head_dim] FP16 (past tokens)
 *   v_cache  : [max_seq_len][n_kv_heads][head_dim] FP16 (past tokens)
 *   output   : [n_tokens][n_q_heads][head_dim] FP32
 * Computes softmax(Q @ K^T / sqrt(head_dim)) @ V for each token in the chunk,
 * attending to all positions up to and including that token (causal). All K/V
 * are read from the FP16 cache, so the caller must write the current chunk into
 * the cache before dispatching this kernel. This keeps the chunk path identical
 * to the token-by-token decode path. This is an intentionally simple O(n*m)
 * implementation, not Flash Attention. */
int ds3_metal_attention_chunk(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k_cache,
    const ds3_metal_buffer_t *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t n_tokens,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim);

/* MoE FFN (FP32 or Q4_K expert weights; FP32 or quantized router).
 * input      : [n_embd]
 * w_gate_inp : [n_expert][n_embd] router weights
 * w_gate_exps: [n_expert][n_ff_exp][n_embd] packed expert gate weights
 * w_up_exps  : [n_expert][n_ff_exp][n_embd] packed expert up weights
 * w_down_exps: [n_expert][n_embd][n_ff_exp] packed expert down weights
 * output        : [n_embd], must be zeroed by caller (accumulates weighted experts)
 * gate_logits   : [n_expert] scratch for router logits (read back to host)
 * expert_offsets: optional [n_expert][3] uint64_t table of byte offsets
 *                 (gate, up, down). When provided and the routed types match
 *                 the GPU-only path (Q4_K gate/up + Q6_K down), the router
 *                 matmul, softmax, top-k and all selected experts are performed
 *                 on the GPU without CPU readback.
 * hidden        : [n_ff_exp] scratch (FP32 path: hidden state;
 *                                   Q4_K path: gate scratch)
 * expert_up     : [n_ff_exp] scratch, only used for Q4_K expert weights
 * expert_down   : [n_embd]   scratch, only used for Q4_K expert weights
 *
 * Router softmax/topk/renorm is done on the CPU after reading gate_logits
 * unless expert_offsets enables the GPU-only path. This function dispatches
 * the router matmul (or the GPU-only kernel), then for each selected expert
 * dispatches gate/up/down kernels and accumulates into output.
 */
int ds3_metal_moe_ffn(
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_inp,
    const ds3_metal_buffer_t *w_gate_exps,
    const ds3_metal_buffer_t *w_up_exps,
    const ds3_metal_buffer_t *w_down_exps,
    ds3_type_t gate_type,
    ds3_type_t up_type,
    ds3_type_t down_type,
    uint64_t gate_row_stride,
    uint64_t up_row_stride,
    uint64_t down_row_stride,
    const ds3_metal_buffer_t *w_shared_gate,
    const ds3_metal_buffer_t *w_shared_up,
    const ds3_metal_buffer_t *w_shared_down,
    ds3_type_t shared_gate_type,
    ds3_type_t shared_up_type,
    ds3_type_t shared_down_type,
    uint64_t shared_gate_row_stride,
    uint64_t shared_up_row_stride,
    uint64_t shared_down_row_stride,
    ds3_metal_buffer_t       *output,
    ds3_metal_buffer_t       *gate_logits,
    const ds3_metal_buffer_t *expert_offsets,
    ds3_metal_buffer_t       *hidden,
    ds3_metal_buffer_t       *expert_up,
    ds3_metal_buffer_t       *expert_down,
    uint32_t n_embd,
    uint32_t n_expert,
    uint32_t n_used,
    uint32_t n_ff_exp,
    uint32_t n_ff_shared,
    uint64_t router_row_stride,
    ds3_type_t router_type,
    bool norm_topk_prob);

/* Lower-level MoE pieces for interleaved multi-layer batching.
 *  - router: just dispatches the gate_inp matmul to produce gate_logits.
 *  - experts: dispatches shared + selected routed experts and accumulates to output.
 * The caller must manage begin_batch/end_batch around these calls. */
int ds3_metal_moe_ffn_router(
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_inp,
    ds3_metal_buffer_t       *gate_logits,
    uint32_t n_embd,
    uint32_t n_expert,
    uint64_t router_row_stride,
    ds3_type_t router_type);

int ds3_metal_moe_ffn_experts(
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_exps,
    const ds3_metal_buffer_t *w_up_exps,
    const ds3_metal_buffer_t *w_down_exps,
    ds3_type_t gate_type,
    ds3_type_t up_type,
    ds3_type_t down_type,
    uint64_t gate_row_stride,
    uint64_t up_row_stride,
    uint64_t down_row_stride,
    const ds3_metal_buffer_t *w_shared_gate,
    const ds3_metal_buffer_t *w_shared_up,
    const ds3_metal_buffer_t *w_shared_down,
    ds3_type_t shared_gate_type,
    ds3_type_t shared_up_type,
    ds3_type_t shared_down_type,
    uint64_t shared_gate_row_stride,
    uint64_t shared_up_row_stride,
    uint64_t shared_down_row_stride,
    ds3_metal_buffer_t       *output,
    ds3_metal_buffer_t       *hidden,
    ds3_metal_buffer_t       *expert_up,
    ds3_metal_buffer_t       *expert_down,
    uint32_t n_embd,
    uint32_t n_used,
    uint32_t n_ff_exp,
    uint32_t n_ff_shared,
    const int32_t            *indices_host,
    const float              *scores_host);

/* GPU softmax + top-k for a batch of router logits.
 * logits:  [n_layers][n_expert]
 * indices: [n_layers][n_used]
 * scores:  [n_layers][n_used] */
int ds3_metal_moe_router_topk_batch(
    const ds3_metal_buffer_t *logits,
    ds3_metal_buffer_t       *indices,
    ds3_metal_buffer_t       *scores,
    uint32_t n_layers,
    uint32_t n_expert,
    uint32_t n_used,
    bool norm_topk_prob);

/* GPU softmax + top-k for a batch of router logits with multiple tokens per layer.
 * logits:  [n_layers][n_tokens][n_expert]
 * indices: [n_layers][n_tokens][n_used]
 * scores:  [n_layers][n_tokens][n_used] */
int ds3_metal_moe_router_topk_batch_tokens(
    const ds3_metal_buffer_t *logits,
    ds3_metal_buffer_t       *indices,
    ds3_metal_buffer_t       *scores,
    uint32_t n_layers,
    uint32_t n_tokens,
    uint32_t n_expert,
    uint32_t n_used,
    bool norm_topk_prob);

/* Command-buffer batching.
 * When a batch is active, dispatch functions append kernels to a single shared
 * command buffer instead of submitting one per call. Use this to merge many
 * small dispatches (e.g. one transformer layer) into fewer GPU submissions.
 *
 * Typical pattern:
 *   ds3_metal_begin_batch();
 *   kernel_a(...); kernel_b(...); ...
 *   ds3_metal_end_batch();   // commits and waits
 */
int  ds3_metal_begin_batch(void);
void ds3_metal_end_batch(void);

/* Wait for the last synchronous command buffer to complete.
 * When using batching, prefer ds3_metal_end_batch() for synchronization. */
void ds3_metal_synchronize(void);

/* Lightweight profiling helpers (enabled via DS3_METAL_PROFILE=1).
 * Totals are accumulated across command buffers; use reset/print from
 * the engine to report per-token or per-generation breakdowns. */
void ds3_metal_profile_reset(void);
void ds3_metal_profile_print(const char *label);

#ifdef __cplusplus
}
#endif

#endif /* DS3_METAL_H */
