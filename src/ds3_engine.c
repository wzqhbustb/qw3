/*
 * ds3_engine.c — End-to-end Qwen3 inference engine (Metal backend)
 *
 * Phase 2.5: single-token decode forward pass.  Correctness first; all
 * per-layer dispatches are issued synchronously.  Residual adds and token
 * embedding lookup happen on the CPU in small chunks to avoid adding new
 * element-wise kernels for this milestone.
 */

#include "ds3.h"
#include "ds3_gguf.h"
#include "ds3_tokenizer.h"
#include "ds3_metal.h"
#include "ds3_reference.h"
#include "ds3_kv_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* Debug globals for chunk-vs-token divergence hunting.
 * attnout_* capture the raw attention output (before the O matmul);
 * postattn_* capture the hidden state right after attention_out + residual;
 * hidden_* capture it after the full FFN block + residual. */
static float *g_debug_xin_token      = NULL;
static float *g_debug_norm_token     = NULL;
static float *g_debug_q_token        = NULL;
static float *g_debug_k_token        = NULL;
static float *g_debug_v_token        = NULL;
static float *g_debug_attnout_token  = NULL;
static float *g_debug_postattn_token = NULL;
static float *g_debug_hidden_token   = NULL;
static int    g_debug_stop_layer     = -1;
static int    g_debug_stop_full_layer = -1;

/* Decode triage logging.  Set DS3_DEBUG_FORWARD=1 to see the hidden state at
 * the start of every forward_token() call and before the final output norm. */
static int    g_debug_forward_calls  = 0;

/* The llama.cpp qwen3moe graph does not include the optional shared-expert
 * branch, so we default to ignoring it to stay reference-compatible.
 * Set DS3_USE_SHARED_EXPERT=1 to enable shared experts if your model/ground
 * truth requires them. */
static bool use_shared_expert(void)
{
    static int cached = -1;
    if (cached < 0) cached = (getenv("DS3_USE_SHARED_EXPERT") != NULL) ? 1 : 0;
    return cached != 0;
}

/* Full definition of the opaque handle declared in ds3.h */
struct ds3_engine {
    ds3_gguf_t    *gguf;
    ds3_weights_t *weights;
    ds3_vocab_t    vocab;

    /* Token embedding — zero-copy view of the quantized mmap; individual rows
     * are dequantized on the CPU into a small temp buffer when needed. */
    ds3_metal_buffer_t *buf_token_embd_view;

    /* Output projection (tied to embedding if weights->output == NULL). */
    ds3_metal_buffer_t *buf_output_weight;
    bool                output_is_tied;

    ds3_metal_buffer_t *buf_output_norm;

    struct {
        ds3_metal_buffer_t *attn_norm;
        ds3_metal_buffer_t *attn_q_norm;
        ds3_metal_buffer_t *attn_k_norm;
        ds3_metal_buffer_t *attn_q;
        ds3_metal_buffer_t *attn_k;
        ds3_metal_buffer_t *attn_v;
        ds3_metal_buffer_t *attn_output;
        ds3_metal_buffer_t *ffn_norm;
        ds3_metal_buffer_t *ffn_gate_inp;
        ds3_metal_buffer_t *ffn_gate_exps;
        ds3_metal_buffer_t *ffn_up_exps;
        ds3_metal_buffer_t *ffn_down_exps;
        ds3_metal_buffer_t *shared_expert_gate;
        ds3_metal_buffer_t *shared_expert_up;
        ds3_metal_buffer_t *shared_expert_down;
    } layer_bufs[DS3_N_LAYER];

    /* KV cache: [max_seq_len][n_kv_heads][head_dim] FP16 per layer. */
    ds3_metal_buffer_t *kv_k[DS3_N_LAYER];
    ds3_metal_buffer_t *kv_v[DS3_N_LAYER];

    /* Scratch buffers. */
    ds3_metal_buffer_t *buf_hidden;
    ds3_metal_buffer_t *buf_residual;
    ds3_metal_buffer_t *buf_norm;
    ds3_metal_buffer_t *buf_q;
    ds3_metal_buffer_t *buf_k;
    ds3_metal_buffer_t *buf_v;
    ds3_metal_buffer_t *buf_attn_out;
    ds3_metal_buffer_t *buf_ffn_out;
    ds3_metal_buffer_t *buf_gate_logits;
    ds3_metal_buffer_t *buf_router_indices;
    ds3_metal_buffer_t *buf_router_scores;
    ds3_metal_buffer_t *buf_expert_offsets[DS3_N_LAYER];
    ds3_metal_buffer_t *buf_moe_hidden;
    ds3_metal_buffer_t *buf_moe_expert_up;
    ds3_metal_buffer_t *buf_moe_expert_down;
    /* Gathered MoE scratch: flat lists of token ids/scores per expert.
     * Sized [n_expert][chunk] for the routed expert gather/scatter kernels. */
    ds3_metal_buffer_t *buf_moe_gather_ids;
    ds3_metal_buffer_t *buf_moe_gather_scores;
    ds3_metal_buffer_t *buf_logits;
    ds3_metal_buffer_t *buf_positions;   /* [chunk] int32 token positions for RoPE */
    ds3_metal_buffer_t *buf_layer_residuals; /* [layers][chunk][n_embd] FP32 */

    float *logits_host;

    /* Zero-copy base buffer for the mmap'd GGUF weight data. */
    ds3_metal_buffer_t *weight_base_buf;

    int n_ctx;
    int seq_len;

    /* KV-cache provider for prefix caching.  NULL means no caching. */
    ds3_kv_cache_provider_t *kv_provider;
    char                     session_id[64];
    void                    *kv_cache_host; /* [n_ctx][2][n_layer][n_kv_head][head_dim] FP16 */
    size_t                   kv_cache_host_bytes;
};

/* ============================================================================
 * Helpers
 * ============================================================================ */

/* Create a zero-copy view of a tensor inside the mmap'd GGUF weight buffer. */
static ds3_metal_buffer_t *upload_weight_view(ds3_engine_t *e, const ds3_tensor_t *t)
{
    if (!e || !t || !e->weight_base_buf || !e->gguf) return NULL;
    size_t offset = (const uint8_t *)t->data - (const uint8_t *)e->gguf->mmap_base;
    return ds3_metal_buffer_view(e->weight_base_buf, offset, t->size);
}

/* Dequantize an entire tensor to FP32 on the host and upload a FP32 copy
 * to the GPU.  If host_out != NULL, the host copy is retained (caller frees
 * later); otherwise it is freed after upload. */
static ds3_metal_buffer_t *upload_dequantized_fp32(const ds3_tensor_t *t,
                                                    float **host_out)
{
    if (!t) return NULL;
    float *host = ds3_ref_dequantize_tensor(t);
    if (!host) {
        fprintf(stderr, "[engine] failed to dequantize %s\n", t->name);
        return NULL;
    }

    size_t n_elements = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n_elements *= (size_t)t->ne[d];
    size_t bytes = n_elements * sizeof(float);

    ds3_metal_buffer_t *gpu = ds3_metal_buffer_alloc(bytes);
    if (!gpu || ds3_metal_buffer_write(gpu, 0, host, bytes) != 0) {
        fprintf(stderr, "[engine] failed to upload dequantized %s\n", t->name);
        free(host);
        ds3_metal_buffer_free(gpu);
        return NULL;
    }

    if (host_out) *host_out = host;
    else          free(host);
    return gpu;
}

static int dispatch_matmul_vec(const ds3_metal_buffer_t *x,
                               const ds3_metal_buffer_t *W,
                               ds3_metal_buffer_t       *y,
                               uint32_t in_dim, uint32_t out_dim,
                               const ds3_tensor_t *t)
{
    uint64_t row_stride = t->nb[1];
    ds3_type_t type = t->type;

    if (type == DS3_TYPE_F32) {
        return ds3_metal_vec_matmul_f32_simd(x, W, y, in_dim, out_dim, row_stride);
    } else if (type == DS3_TYPE_Q4_K) {
        return ds3_metal_vec_matmul_q4k_simd(x, W, y, in_dim, out_dim, row_stride);
    } else if (type == DS3_TYPE_Q8_0) {
        return ds3_metal_vec_matmul_q8_0_simd(x, W, y, in_dim, out_dim, row_stride);
    } else if (type == DS3_TYPE_Q6_K) {
        return ds3_metal_vec_matmul_q6k_simd(x, W, y, in_dim, out_dim, row_stride);
    } else if (type == DS3_TYPE_F16) {
        fprintf(stderr, "[engine] F16 layer weights not supported yet (%s)\n",
                t->name);
        return -1;
    } else {
        fprintf(stderr, "[engine] unsupported weight type %d for %s\n",
                type, t->name);
        return -1;
    }
}

/* Output projection: dispatch the final (n_embd -> n_vocab) matmul using the
 * actual quantized type of the output weight tensor (which may be tied to the
 * token embedding).  This avoids a full FP32 dequantized copy. */
static int dispatch_output_matmul(ds3_engine_t *e,
                                  const ds3_metal_buffer_t *x,
                                  ds3_metal_buffer_t       *y)
{
    ds3_tensor_t *t = e->output_is_tied ? e->weights->token_embd : e->weights->output;
    if (!t) return -1;
    uint64_t row_stride = t->nb[1];
    ds3_type_t type = t->type;

    if (type == DS3_TYPE_F32) {
        return ds3_metal_vec_matmul_f32_simd(x, e->buf_output_weight, y,
                                             DS3_N_EMBD, DS3_N_VOCAB,
                                             row_stride);
    } else if (type == DS3_TYPE_Q4_K) {
        return ds3_metal_vec_matmul_q4k_simd(x, e->buf_output_weight, y,
                                             DS3_N_EMBD, DS3_N_VOCAB,
                                             row_stride);
    } else if (type == DS3_TYPE_Q6_K) {
        return ds3_metal_vec_matmul_q6k_simd(x, e->buf_output_weight, y,
                                             DS3_N_EMBD, DS3_N_VOCAB,
                                             row_stride);
    } else if (type == DS3_TYPE_Q8_0) {
        return ds3_metal_vec_matmul_q8_0_simd(x, e->buf_output_weight, y,
                                              DS3_N_EMBD, DS3_N_VOCAB,
                                              row_stride);
    } else {
        fprintf(stderr, "[engine] unsupported output weight type %d\n", type);
        return -1;
    }
}

/* Batched matrix multiply: C = A @ W^T
 *   A: [M][K]
 *   W: [N][K] (row-major, possibly quantized)
 *   C: [M][N]
 * M is the chunk size; for M == 1 this still works but dispatch_matmul_vec
 * above is tuned for decode. */
static int dispatch_matmul_batch(const ds3_metal_buffer_t *A,
                                 const ds3_metal_buffer_t *W,
                                 ds3_metal_buffer_t       *C,
                                 uint32_t M, uint32_t N, uint32_t K,
                                 const ds3_tensor_t *t)
{
    uint64_t row_stride = t->nb[1];
    ds3_type_t type = t->type;

    if (type == DS3_TYPE_F32) {
        return ds3_metal_matmul(A, W, C, M, N, K);
    } else if (type == DS3_TYPE_Q4_K || type == DS3_TYPE_Q8_0 || type == DS3_TYPE_Q6_K) {
        return ds3_metal_matmul_quantized_batch(A, W, C, M, N, K, row_stride, 0, type);
    } else if (type == DS3_TYPE_F16) {
        fprintf(stderr, "[engine] F16 layer weights not supported yet (%s)\n",
                t->name);
        return -1;
    } else {
        fprintf(stderr, "[engine] unsupported weight type %d for %s\n",
                type, t->name);
        return -1;
    }
}

static int upload_token_embeddings(ds3_engine_t *e, const int *token_ids, int n,
                                   ds3_metal_buffer_t *dst)
{
    if (!e->weights->token_embd) return -1;
    float *tmp = (float *)malloc((size_t)n * DS3_N_EMBD * sizeof(float));
    if (!tmp) return -1;
    for (int i = 0; i < n; i++) {
        if (ds3_ref_dequantize_row(e->weights->token_embd, token_ids[i],
                                   tmp + (size_t)i * DS3_N_EMBD) != 0) {
            free(tmp);
            return -1;
        }
    }
    int rc = ds3_metal_buffer_write(dst, 0, tmp,
                                    (size_t)n * DS3_N_EMBD * sizeof(float));
    free(tmp);
    return rc;
}

static int token_embedding_to_gpu(ds3_engine_t *e, int token_id,
                                  ds3_metal_buffer_t *dst)
{
    return upload_token_embeddings(e, &token_id, 1, dst);
}

static int chunk_embeddings_to_gpu(ds3_engine_t *e, const int *token_ids, int n,
                                   ds3_metal_buffer_t *dst)
{
    return upload_token_embeddings(e, token_ids, n, dst);
}

#define CHECK(x) do { \
    int _rc = (x); \
    if (_rc != 0) { \
        fprintf(stderr, "[engine] %s failed at line %d\n", #x, __LINE__); \
        ds3_metal_end_batch(); \
        return -1; \
    } \
} while (0)

static int sample_token(const float *logits, int n_vocab, float temperature)
{
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    float maxv = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > maxv) maxv = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        sum += expf((logits[i] - maxv) / temperature);
    }

    float r = (float)rand() / (float)RAND_MAX;
    float threshold = r * sum;
    float cum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cum += expf((logits[i] - maxv) / temperature);
        if (cum >= threshold) return i;
    }
    return n_vocab - 1;
}

/* Run the attention branch for a single layer starting from the current
 * contents of e->buf_hidden.  Used by debug/validation code to isolate
 * attention-vs-FFN errors. */
static int run_attention_single_layer(ds3_engine_t *e, int l, int seq_pos)
{
    CHECK(ds3_metal_begin_batch());
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    DS3_N_EMBD * sizeof(float)));
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].attn_norm,
                             e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS));
    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_q, e->buf_q,
                              DS3_N_EMBD, DS3_Q_DIM,
                              e->weights->layers[l].attn_q));
    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_k, e->buf_k,
                              DS3_N_EMBD, DS3_KV_DIM,
                              e->weights->layers[l].attn_k));
    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_v, e->buf_v,
                              DS3_N_EMBD, DS3_KV_DIM,
                              e->weights->layers[l].attn_v));
#if DS3_HAS_QK_NORM
    CHECK(ds3_metal_rms_norm(e->buf_q, e->layer_bufs[l].attn_q_norm, e->buf_q,
                             DS3_HEAD_DIM, DS3_N_HEAD, DS3_NORM_EPS));
    CHECK(ds3_metal_rms_norm(e->buf_k, e->layer_bufs[l].attn_k_norm, e->buf_k,
                             DS3_HEAD_DIM, DS3_N_HEAD_KV, DS3_NORM_EPS));
#endif
    CHECK(ds3_metal_attention_decode_rope_simd(
        e->buf_q, e->buf_k, e->buf_v,
        e->kv_k[l], e->kv_v[l], e->buf_attn_out,
        (uint32_t)seq_pos, (uint32_t)e->n_ctx,
        DS3_N_HEAD, DS3_N_HEAD_KV, DS3_HEAD_DIM,
        DS3_ROPE_THETA));
    CHECK(dispatch_matmul_vec(e->buf_attn_out, e->layer_bufs[l].attn_output,
                              e->buf_hidden, DS3_Q_DIM, DS3_N_EMBD,
                              e->weights->layers[l].attn_output));
    CHECK(ds3_metal_vec_add_inplace(e->buf_hidden, e->buf_residual, DS3_N_EMBD));
    ds3_metal_end_batch();
    return 0;
}

/* Run the FFN branch for a single layer starting from `input`.
 * `indices`/`scores` are the desired router top-k (e.g. from the reference).
 * Returns the MoE output in e->buf_ffn_out. */
static int run_ffn_single_layer(ds3_engine_t *e, int l,
                                const float *input,
                                const int32_t *indices,
                                const float *scores)
{
    CHECK(ds3_metal_buffer_write(e->buf_hidden, 0, input,
                                 DS3_N_EMBD * sizeof(float)));
    CHECK(ds3_metal_begin_batch());
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].ffn_norm,
                             e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS));
    CHECK(ds3_metal_buffer_zero_f32(e->buf_ffn_out,
                                    DS3_N_EMBD * sizeof(float)));

    ds3_layer_weights_t *lw = &e->weights->layers[l];
    const bool has_shared = use_shared_expert() && lw->shared_expert_gate;
    CHECK(ds3_metal_moe_ffn_experts(
        e->buf_norm,
        e->layer_bufs[l].ffn_gate_exps,
        e->layer_bufs[l].ffn_up_exps,
        e->layer_bufs[l].ffn_down_exps,
        lw->ffn_gate_exps->type,
        lw->ffn_up_exps->type,
        lw->ffn_down_exps->type,
        lw->ffn_gate_exps->nb[1],
        lw->ffn_up_exps->nb[1],
        lw->ffn_down_exps->nb[1],
        has_shared ? e->layer_bufs[l].shared_expert_gate : NULL,
        has_shared ? e->layer_bufs[l].shared_expert_up   : NULL,
        has_shared ? e->layer_bufs[l].shared_expert_down : NULL,
        has_shared ? lw->shared_expert_gate->type : DS3_TYPE_F32,
        has_shared ? lw->shared_expert_up->type   : DS3_TYPE_F32,
        has_shared ? lw->shared_expert_down->type : DS3_TYPE_F32,
        has_shared ? lw->shared_expert_gate->nb[1] : 0,
        has_shared ? lw->shared_expert_up->nb[1]   : 0,
        has_shared ? lw->shared_expert_down->nb[1] : 0,
        e->buf_ffn_out,
        e->buf_moe_hidden,
        e->buf_moe_expert_up,
        e->buf_moe_expert_down,
        DS3_N_EMBD, DS3_N_EXPERT_USED, DS3_N_FF_EXP,
        has_shared ? DS3_N_FF_SHARED : 0,
        indices, scores));
    ds3_metal_end_batch();
    return 0;
}

/* Pre-router phase for one layer: attention + router matmul.
 * Writes router logits to e->buf_gate_logits at offset l*DS3_N_EXPERT. */
static int forward_layer_pre(ds3_engine_t *e, int l, int seq_pos)
{
    if (g_debug_xin_token) {
        ds3_metal_buffer_read(e->buf_hidden, 0,
                              g_debug_xin_token + (size_t)l * DS3_N_EMBD,
                              DS3_N_EMBD * sizeof(float));
    }

    /* Save residual before attention branch. */
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    DS3_N_EMBD * sizeof(float)));

    /* --- Attention branch --- */
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].attn_norm,
                             e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS));

    if (g_debug_norm_token) {
        ds3_metal_buffer_read(e->buf_norm, 0,
                              g_debug_norm_token + (size_t)l * DS3_N_EMBD,
                              DS3_N_EMBD * sizeof(float));
    }

    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_q, e->buf_q,
                              DS3_N_EMBD, DS3_Q_DIM,
                              e->weights->layers[l].attn_q));
    if (g_debug_q_token) {
        ds3_metal_buffer_read(e->buf_q, 0,
                              g_debug_q_token + (size_t)l * DS3_Q_DIM,
                              DS3_Q_DIM * sizeof(float));
    }

    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_k, e->buf_k,
                              DS3_N_EMBD, DS3_KV_DIM,
                              e->weights->layers[l].attn_k));
    if (g_debug_k_token) {
        ds3_metal_buffer_read(e->buf_k, 0,
                              g_debug_k_token + (size_t)l * DS3_KV_DIM,
                              DS3_KV_DIM * sizeof(float));
    }

    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_v, e->buf_v,
                              DS3_N_EMBD, DS3_KV_DIM,
                              e->weights->layers[l].attn_v));

    if (g_debug_v_token) {
        ds3_metal_buffer_read(e->buf_v, 0,
                              g_debug_v_token + (size_t)l * DS3_KV_DIM,
                              DS3_KV_DIM * sizeof(float));
    }

#if DS3_HAS_QK_NORM
    CHECK(ds3_metal_rms_norm(e->buf_q, e->layer_bufs[l].attn_q_norm, e->buf_q,
                             DS3_HEAD_DIM, DS3_N_HEAD, DS3_NORM_EPS));
    CHECK(ds3_metal_rms_norm(e->buf_k, e->layer_bufs[l].attn_k_norm, e->buf_k,
                             DS3_HEAD_DIM, DS3_N_HEAD_KV, DS3_NORM_EPS));
#endif

    CHECK(ds3_metal_attention_decode_rope_simd(
        e->buf_q, e->buf_k, e->buf_v,
        e->kv_k[l], e->kv_v[l], e->buf_attn_out,
        (uint32_t)seq_pos, (uint32_t)e->n_ctx,
        DS3_N_HEAD, DS3_N_HEAD_KV, DS3_HEAD_DIM,
        DS3_ROPE_THETA));

    if (g_debug_attnout_token) {
        ds3_metal_buffer_read(e->buf_attn_out, 0,
                              g_debug_attnout_token + (size_t)l * DS3_Q_DIM,
                              DS3_Q_DIM * sizeof(float));
    }

    CHECK(dispatch_matmul_vec(e->buf_attn_out, e->layer_bufs[l].attn_output,
                              e->buf_hidden, DS3_Q_DIM, DS3_N_EMBD,
                              e->weights->layers[l].attn_output));

    /* buf_hidden = attention_out + residual */
    CHECK(ds3_metal_vec_add_inplace(e->buf_hidden, e->buf_residual, DS3_N_EMBD));

    if (g_debug_postattn_token) {
        ds3_metal_buffer_read(e->buf_hidden, 0,
                              g_debug_postattn_token + (size_t)l * DS3_N_EMBD,
                              DS3_N_EMBD * sizeof(float));
    }

    /* Save post-attention residual for the FFN branch. */
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    DS3_N_EMBD * sizeof(float)));

    /* --- FFN branch up to router --- */
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].ffn_norm,
                             e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS));

    CHECK(ds3_metal_buffer_zero_f32(e->buf_ffn_out, DS3_N_EMBD * sizeof(float)));

    ds3_metal_buffer_t *logits_view = ds3_metal_buffer_view(
        e->buf_gate_logits,
        (size_t)l * DS3_N_EXPERT * sizeof(float),
        DS3_N_EXPERT * sizeof(float));
    if (!logits_view) return -1;

    ds3_layer_weights_t *lw = &e->weights->layers[l];
    int rc = ds3_metal_moe_ffn_router(
        e->buf_norm,
        e->layer_bufs[l].ffn_gate_inp,
        logits_view,
        DS3_N_EMBD, DS3_N_EXPERT,
        lw->ffn_gate_inp->nb[1],
        lw->ffn_gate_inp->type);

    ds3_metal_buffer_free(logits_view);
    return rc;
}

/* Post-router phase for one layer: selected experts + residual add.
 * Uses pre-computed indices/scores for this layer. */
static int forward_layer_post(ds3_engine_t *e, int l,
                              const int32_t *indices_host,
                              const float *scores_host)
{
    ds3_layer_weights_t *lw = &e->weights->layers[l];
    const bool has_shared = use_shared_expert() && lw->shared_expert_gate;
    ds3_tensor_t *sg = has_shared ? lw->shared_expert_gate : NULL;
    ds3_tensor_t *su = has_shared ? lw->shared_expert_up   : NULL;
    ds3_tensor_t *sd = has_shared ? lw->shared_expert_down : NULL;

    int rc = ds3_metal_moe_ffn_experts(
        e->buf_norm,
        e->layer_bufs[l].ffn_gate_exps,
        e->layer_bufs[l].ffn_up_exps,
        e->layer_bufs[l].ffn_down_exps,
        lw->ffn_gate_exps->type,
        lw->ffn_up_exps->type,
        lw->ffn_down_exps->type,
        lw->ffn_gate_exps->nb[1],
        lw->ffn_up_exps->nb[1],
        lw->ffn_down_exps->nb[1],
        sg ? e->layer_bufs[l].shared_expert_gate : NULL,
        su ? e->layer_bufs[l].shared_expert_up   : NULL,
        sd ? e->layer_bufs[l].shared_expert_down : NULL,
        sg ? sg->type : DS3_TYPE_F32,
        su ? su->type : DS3_TYPE_F32,
        sd ? sd->type : DS3_TYPE_F32,
        sg ? sg->nb[1] : 0,
        su ? su->nb[1] : 0,
        sd ? sd->nb[1] : 0,
        e->buf_ffn_out,
        e->buf_moe_hidden,
        e->buf_moe_expert_up,
        e->buf_moe_expert_down,
        DS3_N_EMBD, DS3_N_EXPERT_USED, DS3_N_FF_EXP,
        has_shared ? DS3_N_FF_SHARED : 0,
        indices_host, scores_host);
    if (rc != 0) return rc;

    /* buf_hidden = ffn_out + residual */
    CHECK(ds3_metal_vec_add(e->buf_ffn_out, e->buf_residual,
                            e->buf_hidden, DS3_N_EMBD));

    if (g_debug_hidden_token) {
        ds3_metal_buffer_read(e->buf_hidden, 0,
                              g_debug_hidden_token + (size_t)l * DS3_N_EMBD,
                              DS3_N_EMBD * sizeof(float));
    }

    return 0;
}

/* --------------------------------------------------------------------------
 * Layer-internal 1-CB GPU-only MoE decode path.
 *
 * This fuses attention + router + GPU top-k + routed experts into a single
 * command buffer per layer, avoiding the per-layer CPU readback that the
 * pre/post split requires.  It is only used when the layer's weight types
 * match the fused GPU kernel (Q4_K gate/up, Q4_K/Q6_K down, FP32 router).
 * -------------------------------------------------------------------------- */

static bool debug_capture_active(void)
{
    return g_debug_stop_layer >= 0 || g_debug_stop_full_layer >= 0 ||
           g_debug_xin_token || g_debug_norm_token ||
           g_debug_q_token || g_debug_k_token || g_debug_v_token ||
           g_debug_attnout_token || g_debug_postattn_token ||
           g_debug_hidden_token;
}

static bool layer_gpu_only_supported(ds3_engine_t *e, int l)
{
    const ds3_layer_weights_t *lw = &e->weights->layers[l];
    if (!lw->ffn_gate_exps || !lw->ffn_up_exps || !lw->ffn_down_exps ||
        !lw->ffn_gate_inp) {
        return false;
    }
    /* The fused GPU kernels are built for the standard Qwen3-A3B layout. */
    if (lw->ffn_gate_exps->type != DS3_TYPE_Q4_K) return false;
    if (lw->ffn_up_exps->type   != DS3_TYPE_Q4_K) return false;
    if (lw->ffn_down_exps->type != DS3_TYPE_Q4_K &&
        lw->ffn_down_exps->type != DS3_TYPE_Q6_K) return false;
    if (lw->ffn_gate_inp->type  != DS3_TYPE_F32) return false;
    if (DS3_N_EXPERT_USED > 8 || DS3_N_FF_EXP > 768) return false;
    /* Shared expert is not supported by the fused 1-CB path yet. */
    if (lw->shared_expert_gate || lw->shared_expert_up || lw->shared_expert_down) {
        return false;
    }
    return true;
}

static int forward_layer_gpu_only(ds3_engine_t *e, int l, int seq_pos)
{
    ds3_layer_weights_t *lw = &e->weights->layers[l];

    /* Save residual before attention branch. */
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    DS3_N_EMBD * sizeof(float)));

    /* Attention branch. */
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].attn_norm,
                             e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS));
    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_q, e->buf_q,
                              DS3_N_EMBD, DS3_Q_DIM, lw->attn_q));
    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_k, e->buf_k,
                              DS3_N_EMBD, DS3_KV_DIM, lw->attn_k));
    CHECK(dispatch_matmul_vec(e->buf_norm, e->layer_bufs[l].attn_v, e->buf_v,
                              DS3_N_EMBD, DS3_KV_DIM, lw->attn_v));
#if DS3_HAS_QK_NORM
    CHECK(ds3_metal_rms_norm(e->buf_q, e->layer_bufs[l].attn_q_norm, e->buf_q,
                             DS3_HEAD_DIM, DS3_N_HEAD, DS3_NORM_EPS));
    CHECK(ds3_metal_rms_norm(e->buf_k, e->layer_bufs[l].attn_k_norm, e->buf_k,
                             DS3_HEAD_DIM, DS3_N_HEAD_KV, DS3_NORM_EPS));
#endif
    CHECK(ds3_metal_attention_decode_rope_simd(
        e->buf_q, e->buf_k, e->buf_v,
        e->kv_k[l], e->kv_v[l], e->buf_attn_out,
        (uint32_t)seq_pos, (uint32_t)e->n_ctx,
        DS3_N_HEAD, DS3_N_HEAD_KV, DS3_HEAD_DIM,
        DS3_ROPE_THETA));
    CHECK(dispatch_matmul_vec(e->buf_attn_out, e->layer_bufs[l].attn_output,
                              e->buf_hidden, DS3_Q_DIM, DS3_N_EMBD,
                              lw->attn_output));

    /* buf_hidden = attention_out + residual */
    CHECK(ds3_metal_vec_add_inplace(e->buf_hidden, e->buf_residual, DS3_N_EMBD));

    /* Save post-attention residual for the FFN branch. */
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    DS3_N_EMBD * sizeof(float)));

    /* FFN branch: norm, then fused router + experts on the GPU. */
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].ffn_norm,
                             e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS));
    CHECK(ds3_metal_buffer_zero_f32(e->buf_ffn_out, DS3_N_EMBD * sizeof(float)));

    CHECK(ds3_metal_moe_ffn(
        e->buf_norm,
        e->layer_bufs[l].ffn_gate_inp,
        e->layer_bufs[l].ffn_gate_exps,
        e->layer_bufs[l].ffn_up_exps,
        e->layer_bufs[l].ffn_down_exps,
        lw->ffn_gate_exps->type,
        lw->ffn_up_exps->type,
        lw->ffn_down_exps->type,
        lw->ffn_gate_exps->nb[1],
        lw->ffn_up_exps->nb[1],
        lw->ffn_down_exps->nb[1],
        e->layer_bufs[l].shared_expert_gate,
        e->layer_bufs[l].shared_expert_up,
        e->layer_bufs[l].shared_expert_down,
        lw->shared_expert_gate ? lw->shared_expert_gate->type : DS3_TYPE_F32,
        lw->shared_expert_up   ? lw->shared_expert_up->type   : DS3_TYPE_F32,
        lw->shared_expert_down ? lw->shared_expert_down->type : DS3_TYPE_F32,
        lw->shared_expert_gate ? lw->shared_expert_gate->nb[1] : 0,
        lw->shared_expert_up   ? lw->shared_expert_up->nb[1]   : 0,
        lw->shared_expert_down ? lw->shared_expert_down->nb[1] : 0,
        e->buf_ffn_out,
        e->buf_gate_logits,
        e->buf_expert_offsets[l],
        e->buf_moe_hidden,
        e->buf_moe_expert_up,
        e->buf_moe_expert_down,
        DS3_N_EMBD, DS3_N_EXPERT, DS3_N_EXPERT_USED, DS3_N_FF_EXP, DS3_N_FF_SHARED,
        lw->ffn_gate_inp->nb[1], lw->ffn_gate_inp->type,
        DS3_NORM_TOPK_PROB));

    /* buf_hidden = ffn_out + residual */
    CHECK(ds3_metal_vec_add(e->buf_ffn_out, e->buf_residual,
                            e->buf_hidden, DS3_N_EMBD));

    return 0;
}

/* --------------------------------------------------------------------------
 * Chunk-prefill helpers.
 *
 * forward_chunk_pre:  batched attention + router logits for one layer.
 * forward_chunk_post: per-token MoE FFN + residual add for one layer.
 * run_chunk_layer_pre_topk: begin -> pre -> GPU topk -> end -> readback.
 * -------------------------------------------------------------------------- */

static int forward_chunk_pre(ds3_engine_t *e, int l, int M, int seq_pos)
{
    ds3_layer_weights_t *lw = &e->weights->layers[l];

    /* Save residual before attention branch. */
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    (size_t)M * DS3_N_EMBD * sizeof(float)));

    /* Attention branch.
     * We now use the SIMD batched matmul kernels by default: they match the
     * vec kernel to floating-point round-off and are much more efficient for
     * M > 1.  Set DS3_CHUNK_NO_BATCHED_MATMUL=1 to fall back to per-token
     * vec dispatches (correct but slower). */
    const bool batched_matmul = getenv("DS3_CHUNK_NO_BATCHED_MATMUL") == NULL;

    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].attn_norm,
                             e->buf_norm, DS3_N_EMBD, (uint32_t)M, DS3_NORM_EPS));

    if (batched_matmul) {
        CHECK(dispatch_matmul_batch(e->buf_norm, e->layer_bufs[l].attn_q, e->buf_q,
                                    (uint32_t)M, DS3_Q_DIM, DS3_N_EMBD, lw->attn_q));
        CHECK(dispatch_matmul_batch(e->buf_norm, e->layer_bufs[l].attn_k, e->buf_k,
                                    (uint32_t)M, DS3_KV_DIM, DS3_N_EMBD, lw->attn_k));
        CHECK(dispatch_matmul_batch(e->buf_norm, e->layer_bufs[l].attn_v, e->buf_v,
                                    (uint32_t)M, DS3_KV_DIM, DS3_N_EMBD, lw->attn_v));
    } else {
        for (int t = 0; t < M; t++) {
            ds3_metal_buffer_t *in_view = ds3_metal_buffer_view(
                e->buf_norm, (size_t)t * DS3_N_EMBD * sizeof(float),
                DS3_N_EMBD * sizeof(float));
            ds3_metal_buffer_t *q_view = ds3_metal_buffer_view(
                e->buf_q, (size_t)t * DS3_Q_DIM * sizeof(float),
                DS3_Q_DIM * sizeof(float));
            ds3_metal_buffer_t *k_view = ds3_metal_buffer_view(
                e->buf_k, (size_t)t * DS3_KV_DIM * sizeof(float),
                DS3_KV_DIM * sizeof(float));
            ds3_metal_buffer_t *v_view = ds3_metal_buffer_view(
                e->buf_v, (size_t)t * DS3_KV_DIM * sizeof(float),
                DS3_KV_DIM * sizeof(float));
            if (!in_view || !q_view || !k_view || !v_view) {
                ds3_metal_buffer_free(in_view);
                ds3_metal_buffer_free(q_view);
                ds3_metal_buffer_free(k_view);
                ds3_metal_buffer_free(v_view);
                return -1;
            }
            CHECK(dispatch_matmul_vec(in_view, e->layer_bufs[l].attn_q, q_view,
                                      DS3_N_EMBD, DS3_Q_DIM, lw->attn_q));
            CHECK(dispatch_matmul_vec(in_view, e->layer_bufs[l].attn_k, k_view,
                                      DS3_N_EMBD, DS3_KV_DIM, lw->attn_k));
            CHECK(dispatch_matmul_vec(in_view, e->layer_bufs[l].attn_v, v_view,
                                      DS3_N_EMBD, DS3_KV_DIM, lw->attn_v));
            ds3_metal_buffer_free(in_view);
            ds3_metal_buffer_free(q_view);
            ds3_metal_buffer_free(k_view);
            ds3_metal_buffer_free(v_view);
        }
    }

#if DS3_HAS_QK_NORM
    CHECK(ds3_metal_rms_norm(e->buf_q, e->layer_bufs[l].attn_q_norm, e->buf_q,
                             DS3_HEAD_DIM, (uint32_t)(M * DS3_N_HEAD), DS3_NORM_EPS));
    CHECK(ds3_metal_rms_norm(e->buf_k, e->layer_bufs[l].attn_k_norm, e->buf_k,
                             DS3_HEAD_DIM, (uint32_t)(M * DS3_N_HEAD_KV), DS3_NORM_EPS));
#endif

    /* Per-token decode attention with integrated RoPE: exact match to the
     * token-by-token path.  Batched attention_chunk is faster but currently
     * introduces enough numerical drift to flip expert choices. */
    for (int t = 0; t < M; t++) {
        ds3_metal_buffer_t *q_view = ds3_metal_buffer_view(
            e->buf_q, (size_t)t * DS3_Q_DIM * sizeof(float),
            DS3_Q_DIM * sizeof(float));
        ds3_metal_buffer_t *k_view = ds3_metal_buffer_view(
            e->buf_k, (size_t)t * DS3_KV_DIM * sizeof(float),
            DS3_KV_DIM * sizeof(float));
        ds3_metal_buffer_t *v_view = ds3_metal_buffer_view(
            e->buf_v, (size_t)t * DS3_KV_DIM * sizeof(float),
            DS3_KV_DIM * sizeof(float));
        ds3_metal_buffer_t *out_view = ds3_metal_buffer_view(
            e->buf_attn_out, (size_t)t * DS3_Q_DIM * sizeof(float),
            DS3_Q_DIM * sizeof(float));
        if (!q_view || !k_view || !v_view || !out_view) {
            ds3_metal_buffer_free(q_view);
            ds3_metal_buffer_free(k_view);
            ds3_metal_buffer_free(v_view);
            ds3_metal_buffer_free(out_view);
            return -1;
        }
        CHECK(ds3_metal_attention_decode_rope_simd(
            q_view, k_view, v_view,
            e->kv_k[l], e->kv_v[l], out_view,
            (uint32_t)(seq_pos + t), (uint32_t)e->n_ctx,
            DS3_N_HEAD, DS3_N_HEAD_KV, DS3_HEAD_DIM,
            DS3_ROPE_THETA));
        ds3_metal_buffer_free(q_view);
        ds3_metal_buffer_free(k_view);
        ds3_metal_buffer_free(v_view);
        ds3_metal_buffer_free(out_view);
    }

    if (batched_matmul) {
        CHECK(dispatch_matmul_batch(e->buf_attn_out, e->layer_bufs[l].attn_output,
                                    e->buf_hidden,
                                    (uint32_t)M, DS3_N_EMBD, DS3_Q_DIM,
                                    lw->attn_output));
    } else {
        for (int t = 0; t < M; t++) {
            ds3_metal_buffer_t *out_view = ds3_metal_buffer_view(
                e->buf_attn_out, (size_t)t * DS3_Q_DIM * sizeof(float),
                DS3_Q_DIM * sizeof(float));
            ds3_metal_buffer_t *hid_view = ds3_metal_buffer_view(
                e->buf_hidden, (size_t)t * DS3_N_EMBD * sizeof(float),
                DS3_N_EMBD * sizeof(float));
            if (!out_view || !hid_view) {
                ds3_metal_buffer_free(out_view);
                ds3_metal_buffer_free(hid_view);
                return -1;
            }
            CHECK(dispatch_matmul_vec(out_view, e->layer_bufs[l].attn_output,
                                      hid_view, DS3_Q_DIM, DS3_N_EMBD,
                                      lw->attn_output));
            ds3_metal_buffer_free(out_view);
            ds3_metal_buffer_free(hid_view);
        }
    }

    /* buf_hidden = attention_out + residual */
    CHECK(ds3_metal_vec_add_inplace(e->buf_hidden, e->buf_residual,
                                    (uint32_t)(M * DS3_N_EMBD)));

    /* Save post-attention residual for the FFN branch. */
    CHECK(ds3_metal_buffer_copy_f32(e->buf_hidden, e->buf_residual,
                                    (size_t)M * DS3_N_EMBD * sizeof(float)));

    /* FFN norm + router logits. */
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[l].ffn_norm,
                             e->buf_norm, DS3_N_EMBD, (uint32_t)M, DS3_NORM_EPS));

    for (int t = 0; t < M; t++) {
        ds3_metal_buffer_t *norm_view = ds3_metal_buffer_view(
            e->buf_norm, (size_t)t * DS3_N_EMBD * sizeof(float),
            DS3_N_EMBD * sizeof(float));
        ds3_metal_buffer_t *logits_view = ds3_metal_buffer_view(
            e->buf_gate_logits,
            ((size_t)l * M + t) * DS3_N_EXPERT * sizeof(float),
            DS3_N_EXPERT * sizeof(float));
        if (!norm_view || !logits_view) {
            ds3_metal_buffer_free(norm_view);
            ds3_metal_buffer_free(logits_view);
            return -1;
        }
        CHECK(ds3_metal_moe_ffn_router(
            norm_view,
            e->layer_bufs[l].ffn_gate_inp,
            logits_view,
            DS3_N_EMBD, DS3_N_EXPERT,
            lw->ffn_gate_inp->nb[1],
            lw->ffn_gate_inp->type));
        ds3_metal_buffer_free(norm_view);
        ds3_metal_buffer_free(logits_view);
    }

    return 0;
}

static int forward_chunk_post_gathered(ds3_engine_t *e, int l, int M,
                                       const int32_t *indices, const float *scores)
{
    ds3_layer_weights_t *lw = &e->weights->layers[l];

    /* buf_norm holds the FFN input for each token; buf_residual is the
     * post-attention residual saved by forward_chunk_pre. */
    CHECK(ds3_metal_buffer_zero_f32(e->buf_ffn_out,
                                    (size_t)M * DS3_N_EMBD * sizeof(float)));

    /* Shared expert: all tokens use the same weights, so it is a plain
     * batched matmul (no gather).  Keep it separate from the routed gather. */
    const bool has_shared = (lw->shared_expert_gate && lw->shared_expert_up &&
                             lw->shared_expert_down &&
                             e->layer_bufs[l].shared_expert_gate &&
                             e->layer_bufs[l].shared_expert_up &&
                             e->layer_bufs[l].shared_expert_down);
    if (has_shared) {
        CHECK(dispatch_matmul_batch(e->buf_norm, e->layer_bufs[l].shared_expert_gate,
                                    e->buf_moe_hidden,
                                    (uint32_t)M, DS3_N_FF_SHARED, DS3_N_EMBD,
                                    lw->shared_expert_gate));
        CHECK(dispatch_matmul_batch(e->buf_norm, e->layer_bufs[l].shared_expert_up,
                                    e->buf_moe_expert_up,
                                    (uint32_t)M, DS3_N_FF_SHARED, DS3_N_EMBD,
                                    lw->shared_expert_up));
        CHECK(ds3_metal_silu_mul_f32(e->buf_moe_hidden, e->buf_moe_expert_up,
                                     e->buf_moe_expert_up,
                                     (uint32_t)M * DS3_N_FF_SHARED));
        CHECK(dispatch_matmul_batch(e->buf_moe_expert_up, e->layer_bufs[l].shared_expert_down,
                                    e->buf_moe_expert_down,
                                    (uint32_t)M, DS3_N_EMBD, DS3_N_FF_SHARED,
                                    lw->shared_expert_down));
        CHECK(ds3_metal_vec_add_inplace(e->buf_ffn_out, e->buf_moe_expert_down,
                                        (uint32_t)(M * DS3_N_EMBD)));
    }

    /* Build per-expert token lists from the CPU-side router top-k results. */
    int counts[DS3_N_EXPERT];
    memset(counts, 0, sizeof(counts));

    int32_t *ids_host = (int32_t *)malloc((size_t)DS3_N_EXPERT * M * sizeof(int32_t));
    float   *scores_host = (float *)malloc((size_t)DS3_N_EXPERT * M * sizeof(float));
    if (!ids_host || !scores_host) {
        free(ids_host);
        free(scores_host);
        return -1;
    }

    for (int t = 0; t < M; t++) {
        const int32_t *tok_indices = indices + (size_t)t * DS3_N_EXPERT_USED;
        const float   *tok_scores  = scores  + (size_t)t * DS3_N_EXPERT_USED;
        for (int k = 0; k < DS3_N_EXPERT_USED; k++) {
            int ex = tok_indices[k];
            float sc = tok_scores[k];
            int idx = ex * M + counts[ex];
            ids_host[idx] = t;
            scores_host[idx] = sc;
            counts[ex]++;
        }
    }

    int write_rc = ds3_metal_buffer_write(e->buf_moe_gather_ids, 0, ids_host,
                                          (size_t)DS3_N_EXPERT * M * sizeof(int32_t));
    if (write_rc == 0) {
        write_rc = ds3_metal_buffer_write(e->buf_moe_gather_scores, 0, scores_host,
                                          (size_t)DS3_N_EXPERT * M * sizeof(float));
    }
    free(ids_host);
    free(scores_host);
    CHECK(write_rc);

    /* For each active expert, gather the selected token rows, run gate/up/down
     * with batched quantized matmul, then scatter-add the weighted outputs. */
    for (int ex = 0; ex < DS3_N_EXPERT; ex++) {
        int c = counts[ex];
        if (c == 0) continue;

        CHECK(ds3_metal_gather_rows_f32(e->buf_norm, e->buf_moe_gather_ids,
                                        e->buf_moe_expert_down,
                                        DS3_N_EMBD, (uint32_t)c, (uint32_t)(ex * M)));

        const uint64_t gate_off = (uint64_t)ex * lw->ffn_gate_exps->nb[2];
        const uint64_t up_off   = (uint64_t)ex * lw->ffn_up_exps->nb[2];
        const uint64_t down_off = (uint64_t)ex * lw->ffn_down_exps->nb[2];

        CHECK(ds3_metal_matmul_quantized_batch(
            e->buf_moe_expert_down,
            e->layer_bufs[l].ffn_gate_exps,
            e->buf_moe_hidden,
            (uint32_t)c, DS3_N_FF_EXP, DS3_N_EMBD,
            lw->ffn_gate_exps->nb[1],
            gate_off,
            lw->ffn_gate_exps->type));

        CHECK(ds3_metal_matmul_quantized_batch(
            e->buf_moe_expert_down,
            e->layer_bufs[l].ffn_up_exps,
            e->buf_moe_expert_up,
            (uint32_t)c, DS3_N_FF_EXP, DS3_N_EMBD,
            lw->ffn_up_exps->nb[1],
            up_off,
            lw->ffn_up_exps->type));

        CHECK(ds3_metal_silu_mul_f32(e->buf_moe_hidden, e->buf_moe_expert_up,
                                     e->buf_moe_expert_up,
                                     (uint32_t)c * DS3_N_FF_EXP));

        CHECK(ds3_metal_matmul_quantized_batch(
            e->buf_moe_expert_up,
            e->layer_bufs[l].ffn_down_exps,
            e->buf_moe_expert_down,
            (uint32_t)c, DS3_N_EMBD, DS3_N_FF_EXP,
            lw->ffn_down_exps->nb[1],
            down_off,
            lw->ffn_down_exps->type));

        CHECK(ds3_metal_scatter_add_weighted_f32(
            e->buf_moe_expert_down, e->buf_moe_gather_scores, e->buf_moe_gather_ids,
            e->buf_ffn_out,
            DS3_N_EMBD, (uint32_t)c, (uint32_t)(ex * M)));
    }

    /* buf_hidden = ffn_out + residual */
    CHECK(ds3_metal_vec_add(e->buf_ffn_out, e->buf_residual,
                            e->buf_hidden, (uint32_t)(M * DS3_N_EMBD)));
    return 0;
}

static int forward_chunk_post(ds3_engine_t *e, int l, int M,
                              const int32_t *indices, const float *scores)
{
    ds3_layer_weights_t *lw = &e->weights->layers[l];

    /* buf_norm holds the FFN input for each token; buf_residual is the
     * post-attention residual saved by forward_chunk_pre. */
    CHECK(ds3_metal_buffer_zero_f32(e->buf_ffn_out,
                                    (size_t)M * DS3_N_EMBD * sizeof(float)));

    for (int t = 0; t < M; t++) {
        ds3_metal_buffer_t *input_view = ds3_metal_buffer_view(
            e->buf_norm,
            (size_t)t * DS3_N_EMBD * sizeof(float),
            DS3_N_EMBD * sizeof(float));
        ds3_metal_buffer_t *output_view = ds3_metal_buffer_view(
            e->buf_ffn_out,
            (size_t)t * DS3_N_EMBD * sizeof(float),
            DS3_N_EMBD * sizeof(float));
        ds3_metal_buffer_t *hidden_view = ds3_metal_buffer_view(
            e->buf_moe_hidden,
            (size_t)t * DS3_N_FF_SHARED * sizeof(float),
            DS3_N_FF_EXP * sizeof(float));
        ds3_metal_buffer_t *up_view = ds3_metal_buffer_view(
            e->buf_moe_expert_up,
            (size_t)t * DS3_N_FF_SHARED * sizeof(float),
            DS3_N_FF_EXP * sizeof(float));
        ds3_metal_buffer_t *down_view = ds3_metal_buffer_view(
            e->buf_moe_expert_down,
            (size_t)t * DS3_N_EMBD * sizeof(float),
            DS3_N_EMBD * sizeof(float));
        if (!input_view || !output_view || !hidden_view || !up_view || !down_view) {
            ds3_metal_buffer_free(input_view);
            ds3_metal_buffer_free(output_view);
            ds3_metal_buffer_free(hidden_view);
            ds3_metal_buffer_free(up_view);
            ds3_metal_buffer_free(down_view);
            return -1;
        }

        const int32_t *tok_indices = indices + (size_t)t * DS3_N_EXPERT_USED;
        const float   *tok_scores  = scores  + (size_t)t * DS3_N_EXPERT_USED;

        const bool has_shared = use_shared_expert() && lw->shared_expert_gate;
        int rc = ds3_metal_moe_ffn_experts(
            input_view,
            e->layer_bufs[l].ffn_gate_exps,
            e->layer_bufs[l].ffn_up_exps,
            e->layer_bufs[l].ffn_down_exps,
            lw->ffn_gate_exps->type,
            lw->ffn_up_exps->type,
            lw->ffn_down_exps->type,
            lw->ffn_gate_exps->nb[1],
            lw->ffn_up_exps->nb[1],
            lw->ffn_down_exps->nb[1],
            has_shared ? e->layer_bufs[l].shared_expert_gate : NULL,
            has_shared ? e->layer_bufs[l].shared_expert_up   : NULL,
            has_shared ? e->layer_bufs[l].shared_expert_down : NULL,
            has_shared ? lw->shared_expert_gate->type : DS3_TYPE_F32,
            has_shared ? lw->shared_expert_up->type   : DS3_TYPE_F32,
            has_shared ? lw->shared_expert_down->type : DS3_TYPE_F32,
            has_shared ? lw->shared_expert_gate->nb[1] : 0,
            has_shared ? lw->shared_expert_up->nb[1]   : 0,
            has_shared ? lw->shared_expert_down->nb[1] : 0,
            output_view,
            hidden_view,
            up_view,
            down_view,
            DS3_N_EMBD, DS3_N_EXPERT_USED, DS3_N_FF_EXP,
            has_shared ? DS3_N_FF_SHARED : 0,
            tok_indices, tok_scores);

        ds3_metal_buffer_free(input_view);
        ds3_metal_buffer_free(output_view);
        ds3_metal_buffer_free(hidden_view);
        ds3_metal_buffer_free(up_view);
        ds3_metal_buffer_free(down_view);

        if (rc != 0) return -1;
    }

    /* buf_hidden = ffn_out + residual */
    CHECK(ds3_metal_vec_add(e->buf_ffn_out, e->buf_residual,
                            e->buf_hidden, (uint32_t)(M * DS3_N_EMBD)));
    return 0;
}

static int run_chunk_layer_pre_topk(ds3_engine_t *e, int l, int M, int seq_pos,
                                    int32_t *idx_out, float *score_out)
{
    CHECK(ds3_metal_begin_batch());
    CHECK(forward_chunk_pre(e, l, M, seq_pos));

    ds3_metal_buffer_t *logits_view = ds3_metal_buffer_view(
        e->buf_gate_logits,
        (size_t)l * M * DS3_N_EXPERT * sizeof(float),
        (size_t)M * DS3_N_EXPERT * sizeof(float));
    ds3_metal_buffer_t *idx_view = ds3_metal_buffer_view(
        e->buf_router_indices,
        (size_t)l * M * DS3_N_EXPERT_USED * sizeof(int32_t),
        (size_t)M * DS3_N_EXPERT_USED * sizeof(int32_t));
    ds3_metal_buffer_t *score_view = ds3_metal_buffer_view(
        e->buf_router_scores,
        (size_t)l * M * DS3_N_EXPERT_USED * sizeof(float),
        (size_t)M * DS3_N_EXPERT_USED * sizeof(float));
    if (!logits_view || !idx_view || !score_view) {
        ds3_metal_buffer_free(logits_view);
        ds3_metal_buffer_free(idx_view);
        ds3_metal_buffer_free(score_view);
        ds3_metal_end_batch();
        return -1;
    }

    CHECK(ds3_metal_moe_router_topk_batch_tokens(
        logits_view, idx_view, score_view,
        1, (uint32_t)M, DS3_N_EXPERT, DS3_N_EXPERT_USED,
        DS3_NORM_TOPK_PROB));
    ds3_metal_end_batch();

    int rc = 0;
    if (ds3_metal_buffer_read(idx_view, 0, idx_out,
                              (size_t)M * DS3_N_EXPERT_USED * sizeof(int32_t)) != 0 ||
        ds3_metal_buffer_read(score_view, 0, score_out,
                              (size_t)M * DS3_N_EXPERT_USED * sizeof(float)) != 0) {
        rc = -1;
    }

    ds3_metal_buffer_free(logits_view);
    ds3_metal_buffer_free(idx_view);
    ds3_metal_buffer_free(score_view);
    return rc;
}

/* Process a chunk of M prompt tokens with batched matrix multiplies for the
 * attention and projection paths. The MoE FFN is still evaluated per token
 * inside the chunk (a deliberate Phase-4a simplification).
 *
 * The schedule is now layer-sequential (required for residual correctness):
 *   CB 0: pre(0) + topk(0)                       -> readback indices[0]
 *   CB 1: post(0) + pre(1) + topk(1)             -> readback indices[1]
 *   ...
 *   CB L-1: post(L-2) + pre(L-1) + topk(L-1)     -> readback indices[L-1]
 *   CB L: post(L-1) + output_projection
 */
static int forward_chunk(ds3_engine_t *e, const int *token_ids, int M,
                         float temperature)
{
    const int seq_pos = e->seq_len;
    if (seq_pos + M > e->n_ctx) {
        fprintf(stderr, "[engine] chunk exceeds KV cache (seq_pos=%d M=%d n_ctx=%d)\n",
                seq_pos, M, e->n_ctx);
        return -1;
    }
    if (M <= 0) return -1;

    const bool gathered_moe = (getenv("DS3_CHUNK_NO_GATHERED_MOE") == NULL);

    CHECK(chunk_embeddings_to_gpu(e, token_ids, M, e->buf_hidden));

    int32_t *positions = (int32_t *)malloc((size_t)M * sizeof(int32_t));
    if (!positions) return -1;
    for (int i = 0; i < M; i++) positions[i] = seq_pos + i;
    int rc = ds3_metal_buffer_write(e->buf_positions, 0, positions,
                                    (size_t)M * sizeof(int32_t));
    free(positions);
    if (rc != 0) return -1;

    int32_t *indices = (int32_t *)malloc((size_t)DS3_N_LAYER * M * DS3_N_EXPERT_USED * sizeof(int32_t));
    float   *scores  = (float *)malloc((size_t)DS3_N_LAYER * M * DS3_N_EXPERT_USED * sizeof(float));
    if (!indices || !scores) {
        free(indices); free(scores);
        return -1;
    }

    /* Debug snapshot: stop after the attention branch of layer L. */
    if (g_debug_stop_layer >= 0) {
        int stop = g_debug_stop_layer;
        for (int l = 0; l <= stop && l < DS3_N_LAYER; l++) {
            CHECK(run_chunk_layer_pre_topk(e, l, M, seq_pos,
                                           indices + (size_t)l * M * DS3_N_EXPERT_USED,
                                           scores  + (size_t)l * M * DS3_N_EXPERT_USED));
        }
        free(indices);
        free(scores);
        return 0;
    }

    /* Layer 0 attention + router. */
    CHECK(run_chunk_layer_pre_topk(e, 0, M, seq_pos,
                                   indices, scores));

    /* Overlapped layers: previous FFN + current attention + router. */
    for (int l = 1; l < DS3_N_LAYER; l++) {
        CHECK(ds3_metal_begin_batch());
        if (gathered_moe) {
            CHECK(forward_chunk_post_gathered(e, l - 1, M,
                                              indices + (size_t)(l - 1) * M * DS3_N_EXPERT_USED,
                                              scores  + (size_t)(l - 1) * M * DS3_N_EXPERT_USED));
        } else {
            CHECK(forward_chunk_post(e, l - 1, M,
                                     indices + (size_t)(l - 1) * M * DS3_N_EXPERT_USED,
                                     scores  + (size_t)(l - 1) * M * DS3_N_EXPERT_USED));
        }
        CHECK(forward_chunk_pre(e, l, M, seq_pos));

        ds3_metal_buffer_t *logits_view = ds3_metal_buffer_view(
            e->buf_gate_logits,
            (size_t)l * M * DS3_N_EXPERT * sizeof(float),
            (size_t)M * DS3_N_EXPERT * sizeof(float));
        ds3_metal_buffer_t *idx_view = ds3_metal_buffer_view(
            e->buf_router_indices,
            (size_t)l * M * DS3_N_EXPERT_USED * sizeof(int32_t),
            (size_t)M * DS3_N_EXPERT_USED * sizeof(int32_t));
        ds3_metal_buffer_t *score_view = ds3_metal_buffer_view(
            e->buf_router_scores,
            (size_t)l * M * DS3_N_EXPERT_USED * sizeof(float),
            (size_t)M * DS3_N_EXPERT_USED * sizeof(float));
        if (!logits_view || !idx_view || !score_view) {
            ds3_metal_buffer_free(logits_view);
            ds3_metal_buffer_free(idx_view);
            ds3_metal_buffer_free(score_view);
            ds3_metal_end_batch();
            free(indices); free(scores);
            return -1;
        }

        CHECK(ds3_metal_moe_router_topk_batch_tokens(
            logits_view, idx_view, score_view,
            1, (uint32_t)M, DS3_N_EXPERT, DS3_N_EXPERT_USED,
            DS3_NORM_TOPK_PROB));
        ds3_metal_end_batch();

        CHECK(ds3_metal_buffer_read(idx_view, 0,
                                    indices + (size_t)l * M * DS3_N_EXPERT_USED,
                                    (size_t)M * DS3_N_EXPERT_USED * sizeof(int32_t)));
        CHECK(ds3_metal_buffer_read(score_view, 0,
                                    scores  + (size_t)l * M * DS3_N_EXPERT_USED,
                                    (size_t)M * DS3_N_EXPERT_USED * sizeof(float)));

        ds3_metal_buffer_free(logits_view);
        ds3_metal_buffer_free(idx_view);
        ds3_metal_buffer_free(score_view);
    }

    /* Final FFN + output projection. */
    CHECK(ds3_metal_begin_batch());
    if (gathered_moe) {
        CHECK(forward_chunk_post_gathered(e, DS3_N_LAYER - 1, M,
                                          indices + (size_t)(DS3_N_LAYER - 1) * M * DS3_N_EXPERT_USED,
                                          scores  + (size_t)(DS3_N_LAYER - 1) * M * DS3_N_EXPERT_USED));
    } else {
        CHECK(forward_chunk_post(e, DS3_N_LAYER - 1, M,
                                 indices + (size_t)(DS3_N_LAYER - 1) * M * DS3_N_EXPERT_USED,
                                 scores  + (size_t)(DS3_N_LAYER - 1) * M * DS3_N_EXPERT_USED));
    }

    ds3_metal_buffer_t *last_hidden_view = ds3_metal_buffer_view(
        e->buf_hidden,
        (size_t)(M - 1) * DS3_N_EMBD * sizeof(float),
        DS3_N_EMBD * sizeof(float));
    if (!last_hidden_view) {
        ds3_metal_end_batch();
        free(indices); free(scores);
        return -1;
    }
    CHECK(ds3_metal_rms_norm(last_hidden_view, e->buf_output_norm, e->buf_norm,
                             DS3_N_EMBD, 1, DS3_NORM_EPS));
    CHECK(dispatch_output_matmul(e, e->buf_norm, e->buf_logits));
    ds3_metal_buffer_free(last_hidden_view);
    ds3_metal_end_batch();

    free(indices);
    free(scores);

    CHECK(ds3_metal_buffer_read(e->buf_logits, 0, e->logits_host,
                                DS3_N_VOCAB * sizeof(float)));

    return sample_token(e->logits_host, DS3_N_VOCAB, temperature);
}

/* Run attention + router matmul + GPU top-k for one layer and read back the
 * tiny index/score arrays.  This is the natural break point in all correct
 * decode schedules, because the selected experts are needed on the CPU side
 * by the old per-expert dispatch path. */
static int run_layer_pre_topk(ds3_engine_t *e, int l, int seq_pos,
                              int32_t *idx_out, float *score_out)
{
    CHECK(ds3_metal_begin_batch());
    CHECK(forward_layer_pre(e, l, seq_pos));

    ds3_metal_buffer_t *logits_view = ds3_metal_buffer_view(
        e->buf_gate_logits,
        (size_t)l * DS3_N_EXPERT * sizeof(float),
        DS3_N_EXPERT * sizeof(float));
    ds3_metal_buffer_t *idx_view = ds3_metal_buffer_view(
        e->buf_router_indices,
        (size_t)l * DS3_N_EXPERT_USED * sizeof(int32_t),
        DS3_N_EXPERT_USED * sizeof(int32_t));
    ds3_metal_buffer_t *score_view = ds3_metal_buffer_view(
        e->buf_router_scores,
        (size_t)l * DS3_N_EXPERT_USED * sizeof(float),
        DS3_N_EXPERT_USED * sizeof(float));
    if (!logits_view || !idx_view || !score_view) {
        ds3_metal_buffer_free(logits_view);
        ds3_metal_buffer_free(idx_view);
        ds3_metal_buffer_free(score_view);
        ds3_metal_end_batch();
        return -1;
    }

    CHECK(ds3_metal_moe_router_topk_batch(
        logits_view, idx_view, score_view,
        1, DS3_N_EXPERT, DS3_N_EXPERT_USED,
        DS3_NORM_TOPK_PROB));
    ds3_metal_end_batch();

    int rc = 0;
    if (ds3_metal_buffer_read(idx_view, 0, idx_out,
                              DS3_N_EXPERT_USED * sizeof(int32_t)) != 0 ||
        ds3_metal_buffer_read(score_view, 0, score_out,
                              DS3_N_EXPERT_USED * sizeof(float)) != 0) {
        rc = -1;
    }

    ds3_metal_buffer_free(logits_view);
    ds3_metal_buffer_free(idx_view);
    ds3_metal_buffer_free(score_view);
    return rc;
}

static int forward_token(ds3_engine_t *e, int token_id, float temperature)
{
    const int seq_pos = e->seq_len;
    if (seq_pos >= e->n_ctx) {
        fprintf(stderr, "[engine] KV cache full (seq_len=%d >= n_ctx=%d)\n",
                seq_pos, e->n_ctx);
        return -1;
    }
    if (token_id < 0 || token_id >= DS3_N_VOCAB) {
        fprintf(stderr, "[engine] invalid token_id %d\n", token_id);
        return -1;
    }

    CHECK(token_embedding_to_gpu(e, token_id, e->buf_hidden));

    if (getenv("DS3_DEBUG_FORWARD") != NULL) {
        float h[8] = {0};
        ds3_metal_buffer_read(e->buf_hidden, 0, h, sizeof(h));
        int call = g_debug_forward_calls++;
        fprintf(stderr,
                "[fwd %d] tok=%d seq_pos=%d hidden=[%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f]\n",
                call, token_id, seq_pos,
                h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    /* 2-CB per token schedule with GPU router top-k:
     *   CB 0: all 48 layers of attention + router matmul + GPU softmax/top-k
     *   CPU : read 48*8 indices/scores (~768 bytes)
     *   CB 1: all 48 layers of experts + residual adds + final output projection
     * This drops command-buffer flushes from ~49 to 2 per token. */

    int32_t indices[DS3_N_LAYER][DS3_N_EXPERT_USED];
    float   scores [DS3_N_LAYER][DS3_N_EXPERT_USED];

    /* Decide how many layers to run. */
    int pre_layers;
    if (g_debug_stop_full_layer >= 0) {
        pre_layers = g_debug_stop_full_layer + 1;
    } else if (g_debug_stop_layer >= 0) {
        pre_layers = g_debug_stop_layer + 1;
    } else {
        pre_layers = DS3_N_LAYER;
    }

    /* Layer-sequential scheduling is required for correctness: the FFN output of
     * layer L must be available before layer L+1 starts. */
    const bool gpu_only = getenv("DS3_USE_GPU_ONLY_MOE") != NULL;
    const bool overlap  = getenv("DS3_NO_OVERLAP") == NULL;

    if (gpu_only && !debug_capture_active()) {
        /* Fused 1-CB per layer (attention + router + experts, no CPU readback). */
        for (int l = 0; l < pre_layers; l++) {
            CHECK(ds3_metal_begin_batch());
            CHECK(forward_layer_gpu_only(e, l, seq_pos));
            ds3_metal_end_batch();
        }
    } else if (overlap && !debug_capture_active()) {
        /* Overlapped schedule: post(l-1) and pre(l) live in the same CB,
         * cutting flushes from ~97 to ~49 per token while keeping the
         * efficient per-expert SIMD dispatches. */
        CHECK(run_layer_pre_topk(e, 0, seq_pos, indices[0], scores[0]));

        for (int l = 1; l < pre_layers; l++) {
            CHECK(ds3_metal_begin_batch());
            CHECK(forward_layer_post(e, l - 1, indices[l - 1], scores[l - 1]));
            CHECK(forward_layer_pre(e, l, seq_pos));

            ds3_metal_buffer_t *logits_view = ds3_metal_buffer_view(
                e->buf_gate_logits,
                (size_t)l * DS3_N_EXPERT * sizeof(float),
                DS3_N_EXPERT * sizeof(float));
            ds3_metal_buffer_t *idx_view = ds3_metal_buffer_view(
                e->buf_router_indices,
                (size_t)l * DS3_N_EXPERT_USED * sizeof(int32_t),
                DS3_N_EXPERT_USED * sizeof(int32_t));
            ds3_metal_buffer_t *score_view = ds3_metal_buffer_view(
                e->buf_router_scores,
                (size_t)l * DS3_N_EXPERT_USED * sizeof(float),
                DS3_N_EXPERT_USED * sizeof(float));
            if (!logits_view || !idx_view || !score_view) {
                ds3_metal_buffer_free(logits_view);
                ds3_metal_buffer_free(idx_view);
                ds3_metal_buffer_free(score_view);
                ds3_metal_end_batch();
                return -1;
            }
            CHECK(ds3_metal_moe_router_topk_batch(
                logits_view, idx_view, score_view,
                1, DS3_N_EXPERT, DS3_N_EXPERT_USED,
                DS3_NORM_TOPK_PROB));
            ds3_metal_end_batch();

            CHECK(ds3_metal_buffer_read(idx_view, 0, indices[l],
                                        DS3_N_EXPERT_USED * sizeof(int32_t)));
            CHECK(ds3_metal_buffer_read(score_view, 0, scores[l],
                                        DS3_N_EXPERT_USED * sizeof(float)));

            ds3_metal_buffer_free(logits_view);
            ds3_metal_buffer_free(idx_view);
            ds3_metal_buffer_free(score_view);
        }

        /* Final FFN. */
        CHECK(ds3_metal_begin_batch());
        CHECK(forward_layer_post(e, pre_layers - 1,
                                 indices[pre_layers - 1],
                                 scores[pre_layers - 1]));
        ds3_metal_end_batch();
    } else {
        /* Fallback: pre/topk + post as two separate CBs per layer. */
        for (int l = 0; l < pre_layers; l++) {
            if (gpu_only && layer_gpu_only_supported(e, l)) {
                CHECK(ds3_metal_begin_batch());
                CHECK(forward_layer_gpu_only(e, l, seq_pos));
                ds3_metal_end_batch();
                continue;
            }

            CHECK(run_layer_pre_topk(e, l, seq_pos, indices[l], scores[l]));

            if (g_debug_stop_layer < 0) {
                CHECK(ds3_metal_begin_batch());
                CHECK(forward_layer_post(e, l, indices[l], scores[l]));
                ds3_metal_end_batch();
            }
        }
    }

    if (g_debug_stop_layer >= 0 && g_debug_stop_full_layer < 0) {
        /* Debug snapshot after the attention branch: do not run the FFN. */
        return 0;
    }

    if (g_debug_stop_full_layer >= 0) {
        /* The post phase was already run inside the per-layer loop. */
        return 0;
    }

    if (getenv("DS3_DEBUG_FORWARD") != NULL) {
        float h[8] = {0};
        ds3_metal_buffer_read(e->buf_hidden, 0, h, sizeof(h));
        fprintf(stderr,
                "[fwd %d] pre-out-norm hidden=[%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f]\n",
                g_debug_forward_calls - 1,
                h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    /* Output projection. */
    CHECK(ds3_metal_begin_batch());
    CHECK(ds3_metal_rms_norm(e->buf_hidden, e->buf_output_norm, e->buf_hidden,
                             DS3_N_EMBD, 1, DS3_NORM_EPS));
    CHECK(dispatch_output_matmul(e, e->buf_hidden, e->buf_logits));
    ds3_metal_end_batch();

    CHECK(ds3_metal_buffer_read(e->buf_logits, 0, e->logits_host,
                                DS3_N_VOCAB * sizeof(float)));

    if (getenv("DS3_DEBUG_FORWARD") != NULL) {
        int top_i = 0;
        for (int i = 1; i < DS3_N_VOCAB; i++) {
            if (e->logits_host[i] > e->logits_host[top_i]) top_i = i;
        }
        fprintf(stderr, "[fwd %d] sampled=%d top1=%d:%.4f\n",
                g_debug_forward_calls - 1,
                sample_token(e->logits_host, DS3_N_VOCAB, temperature),
                top_i, e->logits_host[top_i]);
    }

    return sample_token(e->logits_host, DS3_N_VOCAB, temperature);
}

#undef CHECK

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

static void free_kv_cache_host(ds3_engine_t *e);

static void free_layer_buffers(ds3_engine_t *e)
{
    for (int l = 0; l < DS3_N_LAYER; l++) {
        ds3_metal_buffer_free(e->layer_bufs[l].attn_norm);
        ds3_metal_buffer_free(e->layer_bufs[l].attn_q_norm);
        ds3_metal_buffer_free(e->layer_bufs[l].attn_k_norm);
        ds3_metal_buffer_free(e->layer_bufs[l].attn_q);
        ds3_metal_buffer_free(e->layer_bufs[l].attn_k);
        ds3_metal_buffer_free(e->layer_bufs[l].attn_v);
        ds3_metal_buffer_free(e->layer_bufs[l].attn_output);
        ds3_metal_buffer_free(e->layer_bufs[l].ffn_norm);
        ds3_metal_buffer_free(e->layer_bufs[l].ffn_gate_inp);
        ds3_metal_buffer_free(e->layer_bufs[l].ffn_gate_exps);
        ds3_metal_buffer_free(e->layer_bufs[l].ffn_up_exps);
        ds3_metal_buffer_free(e->layer_bufs[l].ffn_down_exps);
        ds3_metal_buffer_free(e->layer_bufs[l].shared_expert_gate);
        ds3_metal_buffer_free(e->layer_bufs[l].shared_expert_up);
        ds3_metal_buffer_free(e->layer_bufs[l].shared_expert_down);
        ds3_metal_buffer_free(e->kv_k[l]);
        ds3_metal_buffer_free(e->kv_v[l]);
    }
}

void ds3_engine_close(ds3_engine_t *e)
{
    if (!e) return;

    free_layer_buffers(e);

    ds3_metal_buffer_free(e->weight_base_buf);
    ds3_metal_buffer_free(e->buf_token_embd_view);
    if (!e->output_is_tied) ds3_metal_buffer_free(e->buf_output_weight);
    ds3_metal_buffer_free(e->buf_output_norm);
    ds3_metal_buffer_free(e->buf_hidden);
    ds3_metal_buffer_free(e->buf_residual);
    ds3_metal_buffer_free(e->buf_norm);
    ds3_metal_buffer_free(e->buf_q);
    ds3_metal_buffer_free(e->buf_k);
    ds3_metal_buffer_free(e->buf_v);
    ds3_metal_buffer_free(e->buf_attn_out);
    ds3_metal_buffer_free(e->buf_ffn_out);
    ds3_metal_buffer_free(e->buf_gate_logits);
    ds3_metal_buffer_free(e->buf_router_indices);
    ds3_metal_buffer_free(e->buf_router_scores);
    for (int l = 0; l < DS3_N_LAYER; l++) {
        ds3_metal_buffer_free(e->buf_expert_offsets[l]);
    }
    ds3_metal_buffer_free(e->buf_moe_hidden);
    ds3_metal_buffer_free(e->buf_moe_expert_up);
    ds3_metal_buffer_free(e->buf_moe_expert_down);
    ds3_metal_buffer_free(e->buf_moe_gather_ids);
    ds3_metal_buffer_free(e->buf_moe_gather_scores);
    ds3_metal_buffer_free(e->buf_logits);
    ds3_metal_buffer_free(e->buf_positions);
    ds3_metal_buffer_free(e->buf_layer_residuals);

    free(e->logits_host);
    free_kv_cache_host(e);

    ds3_vocab_free(&e->vocab);
    ds3_weights_free(e->weights);
    ds3_gguf_close(e->gguf);

    free(e);
    ds3_metal_shutdown();
}

static uint64_t tensor_nelements(const ds3_tensor_t *t)
{
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n *= t->ne[d];
    return n;
}

/* Upload a 1-D normalization weight.  RMS-norm kernels expect FP32, so any
 * quantized norm tensor is dequantized and the descriptor is mutated to F32. */
static ds3_metal_buffer_t *upload_norm_weight(ds3_engine_t *e, ds3_tensor_t *t)
{
    if (!t) return NULL;
    if (t->type == DS3_TYPE_F32) return upload_weight_view(e, t);

    ds3_log_info("[engine] dequantizing %s (type=%d) to FP32 for RMS norm\n",
            t->name, t->type);
    ds3_metal_buffer_t *gpu = upload_dequantized_fp32(t, NULL);
    if (!gpu) return NULL;

    t->type = DS3_TYPE_F32;
    t->nb[1] = t->ne[0] * sizeof(float);
    return gpu;
}

/* Upload a weight used in a GPU matmul.  F32, Q4_K, Q6_K and Q8_0 are aliased
 * directly from the mmap'd GGUF; other quantized types are dequantized to FP32
 * on the host first and uploaded into a separate GPU buffer. */
static ds3_metal_buffer_t *upload_layer_weight(ds3_engine_t *e, ds3_tensor_t *t)
{
    if (!t) return NULL;
    if (t->type == DS3_TYPE_F32 || t->type == DS3_TYPE_Q4_K ||
        t->type == DS3_TYPE_Q8_0 || t->type == DS3_TYPE_Q6_K) {
        return upload_weight_view(e, t);
    }
    if (t->type == DS3_TYPE_F16) {
        fprintf(stderr, "[engine] F16 layer weights not supported yet (%s)\n", t->name);
        return NULL;
    }

    ds3_log_info("[engine] dequantizing %s (type=%d) to FP32 for GPU matmul\n",
            t->name, t->type);
    ds3_metal_buffer_t *gpu = upload_dequantized_fp32(t, NULL);
    if (!gpu) return NULL;

    t->type = DS3_TYPE_F32;
    t->nb[1] = t->ne[0] * sizeof(float);
    return gpu;
}

static bool validate_layer_weights(const ds3_layer_weights_t *lw, int layer_idx)
{
    bool ok = true;
#define CHECK2D(t, e0, e1, name) do { \
    if (!lw->t) { ok = false; fprintf(stderr, "[engine] layer %d missing %s\n", layer_idx, name); } \
    else if (lw->t->ne[0] != (uint64_t)(e0) || lw->t->ne[1] != (uint64_t)(e1)) { \
        fprintf(stderr, "[engine] layer %d %s shape mismatch: [%llu,%llu] vs [%d,%d]\n", \
                layer_idx, name, \
                (unsigned long long)lw->t->ne[0], (unsigned long long)lw->t->ne[1], \
                (int)(e0), (int)(e1)); \
        ok = false; \
    } \
} while (0)
#define CHECK_NELEM(t, expected, name) do { \
    if (!lw->t) { ok = false; fprintf(stderr, "[engine] layer %d missing %s\n", layer_idx, name); } \
    else if (tensor_nelements(lw->t) != (uint64_t)(expected)) { \
        fprintf(stderr, "[engine] layer %d %s element count mismatch: %llu vs %llu\n", \
                layer_idx, name, \
                (unsigned long long)tensor_nelements(lw->t), (unsigned long long)(expected)); \
        ok = false; \
    } \
} while (0)

    CHECK2D(attn_norm, DS3_N_EMBD, 1, "attn_norm");
#if DS3_HAS_QK_NORM
    CHECK2D(attn_q_norm, DS3_HEAD_DIM, 1, "attn_q_norm");
    CHECK2D(attn_k_norm, DS3_HEAD_DIM, 1, "attn_k_norm");
#endif
    CHECK2D(attn_q, DS3_N_EMBD, DS3_Q_DIM, "attn_q");
    CHECK2D(attn_k, DS3_N_EMBD, DS3_KV_DIM, "attn_k");
    CHECK2D(attn_v, DS3_N_EMBD, DS3_KV_DIM, "attn_v");
    CHECK2D(attn_output, DS3_Q_DIM, DS3_N_EMBD, "attn_output");
    CHECK2D(ffn_norm, DS3_N_EMBD, 1, "ffn_norm");
    CHECK2D(ffn_gate_inp, DS3_N_EMBD, DS3_N_EXPERT, "ffn_gate_inp");
    CHECK_NELEM(ffn_gate_exps, (uint64_t)DS3_N_EXPERT * DS3_N_FF_EXP * DS3_N_EMBD, "ffn_gate_exps");
    CHECK_NELEM(ffn_up_exps, (uint64_t)DS3_N_EXPERT * DS3_N_FF_EXP * DS3_N_EMBD, "ffn_up_exps");
    CHECK_NELEM(ffn_down_exps, (uint64_t)DS3_N_EXPERT * DS3_N_EMBD * DS3_N_FF_EXP, "ffn_down_exps");

    bool has_shared = lw->shared_expert_gate || lw->shared_expert_up || lw->shared_expert_down;
    if (has_shared) {
        CHECK2D(shared_expert_gate, DS3_N_EMBD, DS3_N_FF_SHARED, "shared_expert_gate");
        CHECK2D(shared_expert_up,   DS3_N_EMBD, DS3_N_FF_SHARED, "shared_expert_up");
        CHECK2D(shared_expert_down, DS3_N_FF_SHARED, DS3_N_EMBD, "shared_expert_down");
    }
#undef CHECK2D
#undef CHECK_NELEM
    return ok;
}

static bool upload_all_weights(ds3_engine_t *e)
{
    for (int l = 0; l < DS3_N_LAYER; l++) {
        ds3_layer_weights_t *lw = &e->weights->layers[l];
        if (!validate_layer_weights(lw, l)) return false;
#define UPLOAD_NORM(field) do { \
    e->layer_bufs[l].field = upload_norm_weight(e, lw->field); \
    if (lw->field && !e->layer_bufs[l].field) { \
        fprintf(stderr, "[engine] failed to upload " #field " for layer %d\n", l); \
        return false; \
    } \
} while (0)
#define UPLOAD_MATMUL(field) do { \
    e->layer_bufs[l].field = upload_layer_weight(e, lw->field); \
    if (lw->field && !e->layer_bufs[l].field) { \
        fprintf(stderr, "[engine] failed to upload " #field " for layer %d\n", l); \
        return false; \
    } \
} while (0)
        UPLOAD_NORM(attn_norm);
        UPLOAD_NORM(attn_q_norm);
        UPLOAD_NORM(attn_k_norm);
        UPLOAD_MATMUL(attn_q);
        UPLOAD_MATMUL(attn_k);
        UPLOAD_MATMUL(attn_v);
        UPLOAD_MATMUL(attn_output);
        UPLOAD_NORM(ffn_norm);
        UPLOAD_MATMUL(ffn_gate_inp);
        UPLOAD_MATMUL(ffn_gate_exps);
        UPLOAD_MATMUL(ffn_up_exps);
        UPLOAD_MATMUL(ffn_down_exps);
        UPLOAD_MATMUL(shared_expert_gate);
        UPLOAD_MATMUL(shared_expert_up);
        UPLOAD_MATMUL(shared_expert_down);

        /* Consistency: if one shared-expert tensor is present, all three must be
         * present (the MoE dispatcher validates uniform type). */
        bool has_shared = (e->layer_bufs[l].shared_expert_gate != NULL);
        if (has_shared != (e->layer_bufs[l].shared_expert_up != NULL) ||
            has_shared != (e->layer_bufs[l].shared_expert_down != NULL)) {
            fprintf(stderr, "[engine] layer %d: incomplete shared expert set\n", l);
            return false;
        }
#undef UPLOAD_NORM
#undef UPLOAD_MATMUL
    }
    return true;
}

static bool upload_expert_offsets(ds3_engine_t *e)
{
    uint64_t *host = (uint64_t *)malloc(3 * DS3_N_EXPERT * sizeof(uint64_t));
    if (!host) return false;

    bool ok = true;
    for (int l = 0; l < DS3_N_LAYER; l++) {
        if (!e->buf_expert_offsets[l]) { ok = false; break; }
        ds3_layer_weights_t *lw = &e->weights->layers[l];
        if (!lw->ffn_gate_exps || !lw->ffn_up_exps || !lw->ffn_down_exps) { ok = false; break; }

        for (int i = 0; i < DS3_N_EXPERT; i++) {
            host[3*i + 0] = (uint64_t)i * DS3_N_FF_EXP * lw->ffn_gate_exps->nb[1];
            host[3*i + 1] = (uint64_t)i * DS3_N_FF_EXP * lw->ffn_up_exps->nb[1];
            host[3*i + 2] = (uint64_t)i * DS3_N_EMBD   * lw->ffn_down_exps->nb[1];
        }

        if (ds3_metal_buffer_write(e->buf_expert_offsets[l], 0, host,
                                   3 * DS3_N_EXPERT * sizeof(uint64_t)) != 0) {
            ok = false;
            break;
        }
    }

    free(host);
    return ok;
}

static bool allocate_kv_cache(ds3_engine_t *e)
{
    const size_t bytes = (size_t)e->n_ctx * DS3_N_HEAD_KV * DS3_HEAD_DIM * sizeof(uint16_t);
    for (int l = 0; l < DS3_N_LAYER; l++) {
        e->kv_k[l] = ds3_metal_buffer_alloc(bytes);
        e->kv_v[l] = ds3_metal_buffer_alloc(bytes);
        if (!e->kv_k[l] || !e->kv_v[l]) {
            fprintf(stderr, "[engine] failed to allocate KV cache for layer %d\n", l);
            return false;
        }
    }
    return true;
}

static bool allocate_scratch(ds3_engine_t *e)
{
    const size_t chunk_embd  = (size_t)DS3_PREFILL_CHUNK_SIZE * DS3_N_EMBD;
    const size_t chunk_qdim  = (size_t)DS3_PREFILL_CHUNK_SIZE * DS3_Q_DIM;
    const size_t chunk_kvdim = (size_t)DS3_PREFILL_CHUNK_SIZE * DS3_KV_DIM;

    e->buf_hidden   = ds3_metal_buffer_alloc(chunk_embd * sizeof(float));
    e->buf_residual = ds3_metal_buffer_alloc(chunk_embd * sizeof(float));
    e->buf_norm     = ds3_metal_buffer_alloc(chunk_embd * sizeof(float));
    e->buf_q        = ds3_metal_buffer_alloc(chunk_qdim * sizeof(float));
    e->buf_k        = ds3_metal_buffer_alloc(chunk_kvdim * sizeof(float));
    e->buf_v        = ds3_metal_buffer_alloc(chunk_kvdim * sizeof(float));
    e->buf_attn_out = ds3_metal_buffer_alloc(chunk_qdim * sizeof(float));
    e->buf_ffn_out  = ds3_metal_buffer_alloc(chunk_embd * sizeof(float));
    e->buf_gate_logits      = ds3_metal_buffer_alloc((size_t)DS3_N_LAYER * DS3_PREFILL_CHUNK_SIZE * DS3_N_EXPERT * sizeof(float));
    e->buf_router_indices   = ds3_metal_buffer_alloc((size_t)DS3_N_LAYER * DS3_PREFILL_CHUNK_SIZE * DS3_N_EXPERT_USED * sizeof(int32_t));
    e->buf_router_scores    = ds3_metal_buffer_alloc((size_t)DS3_N_LAYER * DS3_PREFILL_CHUNK_SIZE * DS3_N_EXPERT_USED * sizeof(float));
    for (int l = 0; l < DS3_N_LAYER; l++) {
        e->buf_expert_offsets[l] = ds3_metal_buffer_alloc(3 * DS3_N_EXPERT * sizeof(uint64_t));
    }
    e->buf_moe_hidden       = ds3_metal_buffer_alloc((size_t)DS3_PREFILL_CHUNK_SIZE * DS3_N_FF_SHARED * sizeof(float));
    e->buf_moe_expert_up  = ds3_metal_buffer_alloc((size_t)DS3_PREFILL_CHUNK_SIZE * DS3_N_FF_SHARED * sizeof(float));
    e->buf_moe_expert_down= ds3_metal_buffer_alloc(chunk_embd * sizeof(float));
    e->buf_moe_gather_ids    = ds3_metal_buffer_alloc((size_t)DS3_N_EXPERT * DS3_PREFILL_CHUNK_SIZE * sizeof(int32_t));
    e->buf_moe_gather_scores = ds3_metal_buffer_alloc((size_t)DS3_N_EXPERT * DS3_PREFILL_CHUNK_SIZE * sizeof(float));
    e->buf_logits   = ds3_metal_buffer_alloc(DS3_N_VOCAB * sizeof(float));
    e->buf_positions= ds3_metal_buffer_alloc(DS3_PREFILL_CHUNK_SIZE * sizeof(int32_t));
    e->buf_layer_residuals = ds3_metal_buffer_alloc((size_t)DS3_N_LAYER * DS3_PREFILL_CHUNK_SIZE * DS3_N_EMBD * sizeof(float));
    e->logits_host  = (float *)malloc(DS3_N_VOCAB * sizeof(float));

    bool offsets_ok = true;
    for (int l = 0; l < DS3_N_LAYER; l++) {
        if (!e->buf_expert_offsets[l]) offsets_ok = false;
    }
    if (!e->buf_hidden || !e->buf_residual || !e->buf_norm || !e->buf_q || !e->buf_k || !e->buf_v ||
        !e->buf_attn_out || !e->buf_ffn_out || !e->buf_gate_logits ||
        !e->buf_router_indices || !e->buf_router_scores || !offsets_ok ||
        !e->buf_moe_hidden || !e->buf_moe_expert_up || !e->buf_moe_expert_down ||
        !e->buf_moe_gather_ids || !e->buf_moe_gather_scores ||
        !e->buf_logits || !e->buf_positions || !e->buf_layer_residuals ||
        !e->logits_host) {
        fprintf(stderr, "[engine] failed to allocate scratch buffers\n");
        return false;
    }
    return true;
}

ds3_engine_t *ds3_engine_open(const char *gguf_path, int n_ctx)
{
    if (n_ctx <= 0) n_ctx = 4096;

    if (ds3_metal_init() != 0) {
        fprintf(stderr, "[engine] Metal init failed\n");
        return NULL;
    }

    ds3_engine_t *e = (ds3_engine_t *)calloc(1, sizeof(ds3_engine_t));
    if (!e) return NULL;
    e->n_ctx = n_ctx;
    e->seq_len = 0;

    e->gguf = ds3_gguf_open(gguf_path);
    if (!e->gguf) {
        fprintf(stderr, "[engine] failed to open %s\n", gguf_path);
        goto fail;
    }

    e->weights = ds3_weights_load(e->gguf);
    if (!e->weights) {
        fprintf(stderr, "[engine] failed to load weights\n");
        goto fail;
    }

    /* Create a single zero-copy GPU alias for the mmap'd weight file.
     * Individual tensors are bound as views with byte offsets. */
    e->weight_base_buf = ds3_metal_buffer_from_mmap(e->gguf->mmap_base,
                                                    e->gguf->file_size);
    if (!e->weight_base_buf) {
        fprintf(stderr, "[engine] failed to create zero-copy weight buffer\n");
        goto fail;
    }

    if (!ds3_vocab_load(&e->vocab, e->gguf)) {
        fprintf(stderr, "[engine] failed to load tokenizer vocabulary\n");
        goto fail;
    }
    if (e->vocab.n_vocab != DS3_N_VOCAB) {
        fprintf(stderr, "[engine] vocab size mismatch: got %d, expected %d\n",
                e->vocab.n_vocab, DS3_N_VOCAB);
        goto fail;
    }

    /* Token embedding: zero-copy view of the quantized mmap. */
    if (!e->weights->token_embd) {
        fprintf(stderr, "[engine] missing token_embd\n");
        goto fail;
    }
    if (e->weights->token_embd->ne[0] != DS3_N_EMBD ||
        e->weights->token_embd->ne[1] != DS3_N_VOCAB) {
        fprintf(stderr, "[engine] token_embd shape mismatch: [%llu,%llu], expected [%d,%d]\n",
                (unsigned long long)e->weights->token_embd->ne[0],
                (unsigned long long)e->weights->token_embd->ne[1],
                DS3_N_EMBD, DS3_N_VOCAB);
        goto fail;
    }
    e->buf_token_embd_view = upload_weight_view(e, e->weights->token_embd);
    if (!e->buf_token_embd_view) {
        fprintf(stderr, "[engine] failed to create token_embd view\n");
        goto fail;
    }

    /* Output projection: zero-copy view of the quantized mmap (may be tied). */
    if (e->weights->output) {
        if (e->weights->output->ne[0] != DS3_N_EMBD ||
            e->weights->output->ne[1] != DS3_N_VOCAB) {
            fprintf(stderr, "[engine] output.weight shape mismatch: [%llu,%llu], expected [%d,%d]\n",
                    (unsigned long long)e->weights->output->ne[0],
                    (unsigned long long)e->weights->output->ne[1],
                    DS3_N_EMBD, DS3_N_VOCAB);
            goto fail;
        }
        e->output_is_tied = false;
        e->buf_output_weight = upload_weight_view(e, e->weights->output);
        if (!e->buf_output_weight) {
            fprintf(stderr, "[engine] failed to create output.weight view\n");
            goto fail;
        }
    } else {
        e->output_is_tied = true;
        e->buf_output_weight = e->buf_token_embd_view;
    }

    if (!e->weights->output_norm) {
        fprintf(stderr, "[engine] missing output_norm\n");
        goto fail;
    }
    if (e->weights->output_norm->ne[0] != DS3_N_EMBD) {
        fprintf(stderr, "[engine] output_norm size mismatch: %llu, expected %d\n",
                (unsigned long long)e->weights->output_norm->ne[0], DS3_N_EMBD);
        goto fail;
    }
    e->buf_output_norm = upload_norm_weight(e, e->weights->output_norm);
    if (!e->buf_output_norm) {
        fprintf(stderr, "[engine] failed to upload output_norm\n");
        goto fail;
    }

    if (!upload_all_weights(e))      goto fail;
    if (!allocate_kv_cache(e))       goto fail;
    if (!allocate_scratch(e))        goto fail;
    if (!upload_expert_offsets(e))   goto fail;

    ds3_print_info("[engine] %s loaded: n_ctx=%d\n", DS3_MODEL_NAME, e->n_ctx);
    return e;

fail:
    ds3_engine_close(e);
    return NULL;
}

/* Debug helper: run the same prompt through the token-by-token decode path and
 * the chunk prefill path, then compare hidden states layer-by-layer and the
 * final logits. This is used to hunt down correctness bugs in forward_chunk().
 *
 * Per-layer hidden-state capture is expensive in memory, so it is only done for
 * prompts up to DS3_PREFILL_CHUNK_SIZE tokens. */
static void debug_compare_chunk_and_token(ds3_engine_t *e,
                                          const int *prompt_tokens,
                                          int n_prompt)
{
    float *logits_token = (float *)malloc(DS3_N_VOCAB * sizeof(float));
    float *logits_chunk = (float *)malloc(DS3_N_VOCAB * sizeof(float));
    if (!logits_token || !logits_chunk) {
        free(logits_token); free(logits_chunk);
        return;
    }

    /* Per-layer post-attention comparison using stop-layer snapshots.
     * The two paths are run only through the attention branch of layer L; the
     * resulting hidden state is read back after the command buffer completes,
     * so the values are trustworthy. */
    float *postattn_token = (float *)malloc((size_t)n_prompt * DS3_N_EMBD * sizeof(float));
    float *postattn_chunk = (float *)malloc((size_t)n_prompt * DS3_N_EMBD * sizeof(float));
    if (!postattn_token || !postattn_chunk) {
        free(postattn_token); free(postattn_chunk);
        free(logits_token); free(logits_chunk);
        return;
    }

    int first_diverge_layer = -1;
    for (int L = 0; L < DS3_N_LAYER; L++) {
        g_debug_stop_layer = L;

        /* Token-by-token path through layer L. */
        e->seq_len = 0;
        for (int i = 0; i < n_prompt; i++) {
            forward_token(e, prompt_tokens[i], 0.0f);
            if (ds3_metal_buffer_read(e->buf_hidden, 0,
                                      postattn_token + (size_t)i * DS3_N_EMBD,
                                      DS3_N_EMBD * sizeof(float)) != 0) {
                goto debug_done;
            }
            if (i < n_prompt - 1) e->seq_len++;
        }

        /* Chunk path through layer L. */
        e->seq_len = 0;
        forward_chunk(e, prompt_tokens, n_prompt, 0.0f);
        if (ds3_metal_buffer_read(e->buf_hidden, 0, postattn_chunk,
                                  (size_t)n_prompt * DS3_N_EMBD * sizeof(float)) != 0) {
            goto debug_done;
        }

        float maxd = 0.0f;
        int max_t = -1, max_d = -1;
        for (int t = 0; t < n_prompt; t++) {
            for (int d = 0; d < DS3_N_EMBD; d++) {
                float diff = fabsf(postattn_token[(size_t)t * DS3_N_EMBD + d] -
                                   postattn_chunk[(size_t)t * DS3_N_EMBD + d]);
                if (diff > maxd) {
                    maxd = diff;
                    max_t = t;
                    max_d = d;
                }
            }
        }
        if (maxd > 1e-4f) {
            first_diverge_layer = L;
            fprintf(stderr,
                    "[debug] post-attn diverge at layer=%d token=%d dim=%d "
                    "diff=%.6e token=%.6f chunk=%.6f\n",
                    L, max_t, max_d, maxd,
                    postattn_token[(size_t)max_t * DS3_N_EMBD + max_d],
                    postattn_chunk[(size_t)max_t * DS3_N_EMBD + max_d]);
            break;
        }
    }
    g_debug_stop_layer = -1;

    if (first_diverge_layer < 0) {
        fprintf(stderr, "[debug] post-attn hidden matches through all %d layers\n", DS3_N_LAYER);
    }

    /* Full forward comparison (final logits). */
    e->seq_len = 0;
    for (int i = 0; i < n_prompt; i++) {
        forward_token(e, prompt_tokens[i], 0.0f);
        if (i < n_prompt - 1) e->seq_len++;
    }
    memcpy(logits_token, e->logits_host, DS3_N_VOCAB * sizeof(float));

    e->seq_len = 0;
    forward_chunk(e, prompt_tokens, n_prompt, 0.0f);
    memcpy(logits_chunk, e->logits_host, DS3_N_VOCAB * sizeof(float));

    float max_diff = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < DS3_N_VOCAB; i++) {
        float d = fabsf(logits_token[i] - logits_chunk[i]);
        if (d > max_diff) {
            max_diff = d;
            max_idx = i;
        }
    }
    fprintf(stderr,
            "[debug] chunk-vs-token logits: n_prompt=%d max_diff=%.6e @ idx=%d "
            "token=%.6f chunk=%.6f\n",
            n_prompt, max_diff, max_idx,
            max_diff > 0.0f ? logits_token[max_idx] : 0.0f,
            max_diff > 0.0f ? logits_chunk[max_idx] : 0.0f);

debug_done:
    g_debug_stop_layer = -1;
    free(postattn_token); free(postattn_chunk);
    free(logits_token); free(logits_chunk);
}

int ds3_engine_debug_layer0_compare(ds3_engine_t *e, int token_id)
{
    if (!e || token_id < 0 || token_id >= DS3_N_VOCAB) return -1;

    float *engine_hidden = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!engine_hidden) return -1;

    /* Run the engine through Layer 0 attention (stops before the FFN). */
    e->seq_len = 0;
    g_debug_stop_layer = 0;
    if (forward_token(e, token_id, 0.0f) != 0) {
        free(engine_hidden);
        g_debug_stop_layer = -1;
        return -1;
    }
    g_debug_stop_layer = -1;

    if (ds3_metal_buffer_read(e->buf_hidden, 0, engine_hidden,
                              DS3_N_EMBD * sizeof(float)) != 0) {
        free(engine_hidden);
        return -1;
    }

    /* FP32 reference for Layer 0 attention. */
    const ds3_layer_weights_t *lw = &e->weights->layers[0];
    float *input = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!input || ds3_ref_dequantize_row(e->weights->token_embd, token_id, input) != 0) {
        free(engine_hidden);
        free(input);
        return -1;
    }

    float *attn_norm   = ds3_ref_dequantize_tensor(lw->attn_norm);
    float *attn_q      = ds3_ref_dequantize_tensor(lw->attn_q);
    float *attn_q_norm = DS3_HAS_QK_NORM ? ds3_ref_dequantize_tensor(lw->attn_q_norm) : NULL;
    float *attn_k      = ds3_ref_dequantize_tensor(lw->attn_k);
    float *attn_k_norm = DS3_HAS_QK_NORM ? ds3_ref_dequantize_tensor(lw->attn_k_norm) : NULL;
    float *attn_v      = ds3_ref_dequantize_tensor(lw->attn_v);
    float *attn_o      = ds3_ref_dequantize_tensor(lw->attn_output);
    if (!attn_norm || !attn_q || (DS3_HAS_QK_NORM && !attn_q_norm) ||
        !attn_k || (DS3_HAS_QK_NORM && !attn_k_norm) ||
        !attn_v || !attn_o) {
        free(engine_hidden);
        free(attn_norm); free(attn_q); free(attn_q_norm);
        free(attn_k); free(attn_k_norm); free(attn_v); free(attn_o);
        return -1;
    }

    float *xb = (float *)malloc(DS3_N_EMBD * sizeof(float));
    float *q  = (float *)calloc(DS3_Q_DIM, sizeof(float));
    float *k  = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *v  = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *ref_attn_out = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *scores = (float *)calloc(DS3_N_HEAD * 1, sizeof(float));
    if (!xb || !q || !k || !v || !ref_attn_out || !scores) {
        free(engine_hidden); free(xb); free(q); free(k); free(v);
        free(ref_attn_out); free(scores);
        free(attn_norm); free(attn_q); free(attn_q_norm);
        free(attn_k); free(attn_k_norm); free(attn_v); free(attn_o);
        return -1;
    }

    ds3_ref_rms_norm(input, attn_norm, DS3_NORM_EPS, DS3_N_EMBD, xb);

    ds3_kv_cache_t kv_cache = {0};
    kv_cache.k_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
    kv_cache.v_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
    kv_cache.capacity = 1;
    kv_cache.seq_len  = 0;
    if (!kv_cache.k_cache || !kv_cache.v_cache) {
        free(kv_cache.k_cache); free(kv_cache.v_cache);
        free(engine_hidden); free(xb); free(q); free(k); free(v);
        free(ref_attn_out); free(scores);
        free(attn_norm); free(attn_q); free(attn_q_norm);
        free(attn_k); free(attn_k_norm); free(attn_v); free(attn_o);
        return -1;
    }

    ds3_ref_gqa_attention(xb,
                          attn_q, attn_q_norm,
                          attn_k, attn_k_norm,
                          attn_v, attn_o,
                          &kv_cache, 0, 1,
                          ref_attn_out,
                          q, k, v, scores);

    float max_diff = 0.0f;
    int max_d = -1;
    for (int d = 0; d < DS3_N_EMBD; d++) {
        float ref_val = input[d] + ref_attn_out[d];
        float diff = fabsf(engine_hidden[d] - ref_val);
        if (diff > max_diff) {
            max_diff = diff;
            max_d = d;
        }
    }

    fprintf(stderr,
            "[layer0 compare] token_id=%d max_diff=%.6e @ dim=%d "
            "engine=%.6f ref=%.6f\n",
            token_id, max_diff, max_d,
            engine_hidden[max_d], input[max_d] + ref_attn_out[max_d]);

    free(kv_cache.k_cache); free(kv_cache.v_cache);
    free(engine_hidden); free(input); free(xb); free(q); free(k); free(v);
    free(ref_attn_out); free(scores);
    free(attn_norm); free(attn_q); free(attn_q_norm);
    free(attn_k); free(attn_k_norm); free(attn_v); free(attn_o);

    return (max_diff < 1e-3f) ? 0 : 1;
}

int ds3_engine_debug_layer0_full_compare(ds3_engine_t *e, int token_id)
{
    if (!e || token_id < 0 || token_id >= DS3_N_VOCAB) return -1;

    float *engine_hidden = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!engine_hidden) return -1;

    /* Run the engine through Layer 0 attention + FFN. */
    e->seq_len = 0;
    g_debug_stop_full_layer = 0;
    if (forward_token(e, token_id, 0.0f) != 0) {
        free(engine_hidden);
        g_debug_stop_full_layer = -1;
        return -1;
    }
    g_debug_stop_full_layer = -1;

    if (ds3_metal_buffer_read(e->buf_hidden, 0, engine_hidden,
                              DS3_N_EMBD * sizeof(float)) != 0) {
        free(engine_hidden);
        return -1;
    }

    /* FP32 reference for full Layer 0. */
    const ds3_layer_weights_t *lw = &e->weights->layers[0];
    float *input = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!input || ds3_ref_dequantize_row(e->weights->token_embd, token_id, input) != 0) {
        free(engine_hidden);
        free(input);
        return -1;
    }

    ds3_layer_state_t state = {0};
    float *xb          = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *q           = (float *)calloc(DS3_Q_DIM, sizeof(float));
    float *k           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *v           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *attn_out    = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *ffn_out     = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *gate_logits = (float *)calloc(DS3_N_EXPERT, sizeof(float));
    float *exp_scores  = (float *)calloc(DS3_N_EXPERT_USED, sizeof(float));
    int   *exp_indices = (int *)calloc(DS3_N_EXPERT_USED, sizeof(int));
    float *attn_score  = (float *)calloc(DS3_N_HEAD * 1, sizeof(float));
    float *ref_output  = (float *)calloc(DS3_N_EMBD, sizeof(float));

    if (!xb || !q || !k || !v || !attn_out || !ffn_out ||
        !gate_logits || !exp_scores || !exp_indices || !attn_score || !ref_output) {
        free(engine_hidden);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score); free(ref_output);
        return -1;
    }

    state.xb = xb;
    state.q = q; state.k = k; state.v = v;
    state.attn_out = attn_out; state.ffn_out = ffn_out;
    state.gate_logits = gate_logits;
    state.expert_weights = exp_scores;
    state.expert_indices = exp_indices;
    state.attn_score = attn_score;

    ds3_kv_cache_t kv_cache = {0};
    kv_cache.k_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
    kv_cache.v_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
    kv_cache.capacity = 1;
    kv_cache.seq_len  = 0;

    if (!kv_cache.k_cache || !kv_cache.v_cache) {
        free(kv_cache.k_cache); free(kv_cache.v_cache);
        free(engine_hidden);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score); free(ref_output);
        return -1;
    }

    ds3_ref_layer_forward(lw, input, &kv_cache, 0, 1, ref_output, &state);

    float max_diff = 0.0f;
    int max_d = -1;
    for (int d = 0; d < DS3_N_EMBD; d++) {
        float diff = fabsf(engine_hidden[d] - ref_output[d]);
        if (diff > max_diff) {
            max_diff = diff;
            max_d = d;
        }
    }

    fprintf(stderr,
            "[layer0 full compare] token_id=%d max_diff=%.6e @ dim=%d "
            "engine=%.6f ref=%.6f\n",
            token_id, max_diff, max_d,
            max_d >= 0 ? engine_hidden[max_d] : 0.0f,
            max_d >= 0 ? ref_output[max_d] : 0.0f);

    free(kv_cache.k_cache); free(kv_cache.v_cache);
    free(engine_hidden); free(input);
    free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
    free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score); free(ref_output);

    return (max_diff < 1e-3f) ? 0 : 1;
}

int ds3_engine_debug_full_compare(ds3_engine_t *e, int token_id)
{
    if (!e || token_id < 0 || token_id >= DS3_N_VOCAB) return -1;

    /* Engine forward for a single token. */
    e->seq_len = 0;
    if (forward_token(e, token_id, 0.0f) < 0) return -1;

    float *engine_logits = (float *)malloc(DS3_N_VOCAB * sizeof(float));
    if (!engine_logits) return -1;
    memcpy(engine_logits, e->logits_host, DS3_N_VOCAB * sizeof(float));

    /* FP32 reference forward through all layers. */
    float *input = (float *)malloc(DS3_N_EMBD * sizeof(float));
    float *cur = (float *)malloc(DS3_N_EMBD * sizeof(float));
    float *next = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!input || !cur || !next ||
        ds3_ref_dequantize_row(e->weights->token_embd, token_id, input) != 0) {
        free(engine_logits); free(input); free(cur); free(next);
        return -1;
    }
    memcpy(cur, input, DS3_N_EMBD * sizeof(float));

    ds3_layer_state_t state = {0};
    float *xb          = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *q           = (float *)calloc(DS3_Q_DIM, sizeof(float));
    float *k           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *v           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *attn_out    = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *ffn_out     = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *gate_logits = (float *)calloc(DS3_N_EXPERT, sizeof(float));
    float *exp_scores  = (float *)calloc(DS3_N_EXPERT_USED, sizeof(float));
    int   *exp_indices = (int *)calloc(DS3_N_EXPERT_USED, sizeof(int));
    float *attn_score  = (float *)calloc(DS3_N_HEAD * 1, sizeof(float));

    if (!xb || !q || !k || !v || !attn_out || !ffn_out ||
        !gate_logits || !exp_scores || !exp_indices || !attn_score) {
        free(engine_logits); free(cur); free(next);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }

    state.xb = xb;
    state.q = q; state.k = k; state.v = v;
    state.attn_out = attn_out; state.ffn_out = ffn_out;
    state.gate_logits = gate_logits;
    state.expert_weights = exp_scores;
    state.expert_indices = exp_indices;
    state.attn_score = attn_score;

    for (int l = 0; l < DS3_N_LAYER; l++) {
        ds3_kv_cache_t kv_cache = {0};
        kv_cache.k_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
        kv_cache.v_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
        kv_cache.capacity = 1;
        kv_cache.seq_len  = 0;
        if (!kv_cache.k_cache || !kv_cache.v_cache) {
            free(kv_cache.k_cache); free(kv_cache.v_cache);
            break;
        }

        ds3_ref_layer_forward(&e->weights->layers[l], cur, &kv_cache, 0, 1, next, &state);

        free(kv_cache.k_cache);
        free(kv_cache.v_cache);

        float *tmp = cur; cur = next; next = tmp;
    }

    /* Output normalization and projection. */
    float *output_norm_w = ds3_ref_dequantize_tensor(e->weights->output_norm);
    float *normed = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *ref_logits = (float *)calloc(DS3_N_VOCAB, sizeof(float));
    if (!output_norm_w || !normed || !ref_logits) {
        free(engine_logits); free(cur); free(next);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        free(output_norm_w); free(normed); free(ref_logits);
        return -1;
    }

    ds3_ref_rms_norm(cur, output_norm_w, DS3_NORM_EPS, DS3_N_EMBD, normed);

    ds3_tensor_t *output_t = e->output_is_tied ? e->weights->token_embd
                                                 : e->weights->output;
    float *output_w = ds3_ref_dequantize_tensor(output_t);
    if (!output_w) {
        free(engine_logits); free(input); free(cur); free(next);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        free(output_norm_w); free(normed); free(ref_logits);
        return -1;
    }
    ds3_ref_vec_matmul(normed, output_w, ref_logits,
                       DS3_N_EMBD, DS3_N_VOCAB);
    free(output_w);

    float max_diff = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < DS3_N_VOCAB; i++) {
        float diff = fabsf(engine_logits[i] - ref_logits[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    fprintf(stderr,
            "[full compare] token_id=%d max_diff=%.6e @ idx=%d "
            "engine=%.6f ref=%.6f\n",
            token_id, max_diff, max_idx,
            max_idx >= 0 ? engine_logits[max_idx] : 0.0f,
            max_idx >= 0 ? ref_logits[max_idx] : 0.0f);

    free(engine_logits); free(input); free(cur); free(next);
    free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
    free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
    free(output_norm_w); free(normed); free(ref_logits);

    return (max_diff < 1e-3f) ? 0 : 1;
}

/* --------------------------------------------------------------------------
 * Multi-step decode compare.
 *
 * Runs the engine through `tokens[0..n_steps-1]` one token at a time, keeping
 * the KV cache live between steps exactly like real generation, and compares
 * the final logits after each step against the FP32 CPU reference.
 *
 * This is the primary diagnostic for the multi-token decode drift bug: the
 * single-token full_compare passes, but small state errors accumulate here.
 *
 * Implementation note: we run the engine first, then compute the reference
 * layer-by-layer with per-layer weights cached across steps.  This avoids the
 * O(n_steps * n_layers) full-model dequantization of the naive loop and makes
 * the tool fast enough to use interactively.
 * -------------------------------------------------------------------------- */
int ds3_engine_debug_decode_compare(ds3_engine_t *e, const int *tokens, int n_steps)
{
    if (!e || !tokens || n_steps <= 0) return -1;
    if (n_steps > e->n_ctx) {
        fprintf(stderr, "[decode compare] n_steps=%d exceeds n_ctx=%d\n",
                n_steps, e->n_ctx);
        return -1;
    }

    e->seq_len = 0;

    /* Capture engine hidden states and logits for every step. */
    float *engine_hidden = (float *)calloc((size_t)n_steps * DS3_N_EMBD, sizeof(float));
    float *engine_logits = (float *)calloc((size_t)n_steps * DS3_N_VOCAB, sizeof(float));
    if (!engine_hidden || !engine_logits) {
        free(engine_hidden); free(engine_logits);
        return -1;
    }

    for (int step = 0; step < n_steps; step++) {
        if (forward_token(e, tokens[step], 0.0f) < 0) {
            fprintf(stderr, "[decode compare] engine forward failed at step %d\n", step);
            free(engine_hidden); free(engine_logits);
            return -1;
        }
        e->seq_len++;

        if (ds3_metal_buffer_read(e->buf_hidden, 0,
                                  engine_hidden + (size_t)step * DS3_N_EMBD,
                                  DS3_N_EMBD * sizeof(float)) != 0) {
            fprintf(stderr, "[decode compare] failed to read hidden at step %d\n", step);
            free(engine_hidden); free(engine_logits);
            return -1;
        }
        memcpy(engine_logits + (size_t)step * DS3_N_VOCAB,
               e->logits_host, DS3_N_VOCAB * sizeof(float));
    }

    /* Reference scratch buffers. */
    ds3_layer_state_t state = {0};
    float *xb          = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *q           = (float *)calloc(DS3_Q_DIM, sizeof(float));
    float *k           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *v           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *attn_out    = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *ffn_out     = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *gate_logits = (float *)calloc(DS3_N_EXPERT, sizeof(float));
    float *exp_scores  = (float *)calloc(DS3_N_EXPERT_USED, sizeof(float));
    int   *exp_indices = (int *)calloc(DS3_N_EXPERT_USED, sizeof(int));
    float *attn_score  = (float *)calloc((size_t)DS3_N_HEAD * n_steps, sizeof(float));

    if (!xb || !q || !k || !v || !attn_out || !ffn_out ||
        !gate_logits || !exp_scores || !exp_indices || !attn_score) {
        free(engine_hidden); free(engine_logits);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }

    state.xb = xb;
    state.q = q; state.k = k; state.v = v;
    state.attn_out = attn_out; state.ffn_out = ffn_out;
    state.gate_logits = gate_logits;
    state.expert_weights = exp_scores;
    state.expert_indices = exp_indices;
    state.attn_score = attn_score;

    /* Per-layer KV caches, persistent across steps. */
    ds3_kv_cache_t *kv_cache = (ds3_kv_cache_t *)calloc(DS3_N_LAYER, sizeof(ds3_kv_cache_t));
    if (!kv_cache) {
        free(engine_hidden); free(engine_logits);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }
    for (int l = 0; l < DS3_N_LAYER; l++) {
        kv_cache[l].k_cache = (float *)calloc((size_t)n_steps * DS3_N_HEAD_KV * DS3_HEAD_DIM,
                                              sizeof(float));
        kv_cache[l].v_cache = (float *)calloc((size_t)n_steps * DS3_N_HEAD_KV * DS3_HEAD_DIM,
                                              sizeof(float));
        kv_cache[l].capacity = (size_t)n_steps;
        kv_cache[l].seq_len  = 0;
        if (!kv_cache[l].k_cache || !kv_cache[l].v_cache) {
            for (int j = 0; j <= l; j++) {
                free(kv_cache[j].k_cache); free(kv_cache[j].v_cache);
            }
            free(kv_cache);
            free(engine_hidden); free(engine_logits);
            free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
            free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
            return -1;
        }
    }

    /* Output projection weights. */
    float *output_norm_w = ds3_ref_dequantize_tensor(e->weights->output_norm);
    ds3_tensor_t *output_t = e->output_is_tied ? e->weights->token_embd : e->weights->output;
    float *output_w = ds3_ref_dequantize_tensor(output_t);
    if (!output_norm_w || !output_w) {
        free(engine_hidden); free(engine_logits);
        free(output_norm_w); free(output_w);
        for (int l = 0; l < DS3_N_LAYER; l++) {
            free(kv_cache[l].k_cache); free(kv_cache[l].v_cache);
        }
        free(kv_cache);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }

    /* layer_out[(l * n_steps + step) * N_EMBD] = hidden after layer l. */
    float *layer_out = (float *)calloc((size_t)(DS3_N_LAYER + 1) * n_steps * DS3_N_EMBD,
                                       sizeof(float));
    if (!layer_out) {
        free(engine_hidden); free(engine_logits);
        free(output_norm_w); free(output_w);
        for (int l = 0; l < DS3_N_LAYER; l++) {
            free(kv_cache[l].k_cache); free(kv_cache[l].v_cache);
        }
        free(kv_cache);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }

    /* Layer 0 inputs are token embeddings. */
    for (int step = 0; step < n_steps; step++) {
        if (ds3_ref_dequantize_row(e->weights->token_embd, tokens[step],
                                   layer_out + (size_t)step * DS3_N_EMBD) != 0) {
            fprintf(stderr, "[decode compare] embedding lookup failed at step %d\n", step);
            free(layer_out);
            free(engine_hidden); free(engine_logits);
            free(output_norm_w); free(output_w);
            for (int l = 0; l < DS3_N_LAYER; l++) {
                free(kv_cache[l].k_cache); free(kv_cache[l].v_cache);
            }
            free(kv_cache);
            free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
            free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
            return -1;
        }
    }

    /* Reference forward: layer by layer, weights cached across steps. */
    for (int l = 0; l < DS3_N_LAYER; l++) {
        ds3_layer_weights_t *lw = &e->weights->layers[l];

        const float *attn_norm   = ds3_ref_dequantize_tensor(lw->attn_norm);
        const float *attn_q      = ds3_ref_dequantize_tensor(lw->attn_q);
        const float *attn_q_norm = DS3_HAS_QK_NORM ? ds3_ref_dequantize_tensor(lw->attn_q_norm) : NULL;
        const float *attn_k      = ds3_ref_dequantize_tensor(lw->attn_k);
        const float *attn_k_norm = DS3_HAS_QK_NORM ? ds3_ref_dequantize_tensor(lw->attn_k_norm) : NULL;
        const float *attn_v      = ds3_ref_dequantize_tensor(lw->attn_v);
        const float *attn_o      = ds3_ref_dequantize_tensor(lw->attn_output);
        const float *ffn_norm    = ds3_ref_dequantize_tensor(lw->ffn_norm);
        const float *gate_inp    = ds3_ref_dequantize_tensor(lw->ffn_gate_inp);
        const float *gate_exps   = ds3_ref_dequantize_tensor(lw->ffn_gate_exps);
        const float *up_exps     = ds3_ref_dequantize_tensor(lw->ffn_up_exps);
        const float *down_exps   = ds3_ref_dequantize_tensor(lw->ffn_down_exps);
        const bool has_shared = use_shared_expert() && lw->shared_expert_gate;
        const float *sh_gate     = has_shared ? ds3_ref_dequantize_tensor(lw->shared_expert_gate) : NULL;
        const float *sh_up       = has_shared ? ds3_ref_dequantize_tensor(lw->shared_expert_up)   : NULL;
        const float *sh_down     = has_shared ? ds3_ref_dequantize_tensor(lw->shared_expert_down) : NULL;

        if (!attn_norm || !attn_q || !attn_k || !attn_v || !attn_o ||
            !ffn_norm || !gate_inp || !gate_exps || !up_exps || !down_exps ||
            (DS3_HAS_QK_NORM && (!attn_q_norm || !attn_k_norm))) {
            fprintf(stderr, "[decode compare] failed to dequantize weights for layer %d\n", l);
            free((void *)attn_norm); free((void *)attn_q); free((void *)attn_q_norm);
            free((void *)attn_k); free((void *)attn_k_norm); free((void *)attn_v);
            free((void *)attn_o); free((void *)ffn_norm); free((void *)gate_inp);
            free((void *)gate_exps); free((void *)up_exps); free((void *)down_exps);
            free((void *)sh_gate); free((void *)sh_up); free((void *)sh_down);
            free(layer_out);
            free(engine_hidden); free(engine_logits);
            free(output_norm_w); free(output_w);
            for (int j = 0; j < DS3_N_LAYER; j++) {
                free(kv_cache[j].k_cache); free(kv_cache[j].v_cache);
            }
            free(kv_cache);
            free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
            free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
            return -1;
        }

        for (int step = 0; step < n_steps; step++) {
            const float *in  = layer_out + ((size_t)l * n_steps + step) * DS3_N_EMBD;
            float       *out = layer_out + ((size_t)(l + 1) * n_steps + step) * DS3_N_EMBD;

            ds3_ref_rms_norm(in, attn_norm, DS3_NORM_EPS, DS3_N_EMBD, state.xb);

            ds3_ref_gqa_attention(
                state.xb,
                attn_q, attn_q_norm,
                attn_k, attn_k_norm,
                attn_v,
                attn_o,
                &kv_cache[l], step, 1,
                state.attn_out,
                state.q, state.k, state.v,
                state.attn_score);

            ds3_ref_add(in, state.attn_out, state.attn_out, DS3_N_EMBD);
            ds3_ref_rms_norm(state.attn_out, ffn_norm, DS3_NORM_EPS, DS3_N_EMBD, state.xb);

            ds3_ref_moe_ffn(
                state.xb,
                gate_inp, gate_exps, up_exps, down_exps,
                sh_gate, sh_up, sh_down,
                state.ffn_out,
                state.gate_logits,
                state.expert_weights,
                state.expert_indices,
                DS3_N_EMBD, DS3_N_EXPERT, DS3_N_EXPERT_USED,
                DS3_N_FF_EXP, DS3_N_FF_SHARED,
                DS3_NORM_TOPK_PROB);

            ds3_ref_add(state.attn_out, state.ffn_out, out, DS3_N_EMBD);
        }

        free((void *)attn_norm); free((void *)attn_q); free((void *)attn_q_norm);
        free((void *)attn_k); free((void *)attn_k_norm); free((void *)attn_v);
        free((void *)attn_o); free((void *)ffn_norm); free((void *)gate_inp);
        free((void *)gate_exps); free((void *)up_exps); free((void *)down_exps);
        free((void *)sh_gate); free((void *)sh_up); free((void *)sh_down);
    }

    /* Compare engine vs reference for each step. */
    float *normed = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *ref_logits = (float *)calloc(DS3_N_VOCAB, sizeof(float));
    if (!normed || !ref_logits) {
        free(normed); free(ref_logits);
        free(layer_out);
        free(engine_hidden); free(engine_logits);
        free(output_norm_w); free(output_w);
        for (int l = 0; l < DS3_N_LAYER; l++) {
            free(kv_cache[l].k_cache); free(kv_cache[l].v_cache);
        }
        free(kv_cache);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }

    int first_bad_step = -1;
    for (int step = 0; step < n_steps; step++) {
        const float *final_hidden = layer_out + ((size_t)DS3_N_LAYER * n_steps + step) * DS3_N_EMBD;
        ds3_ref_rms_norm(final_hidden, output_norm_w, DS3_NORM_EPS, DS3_N_EMBD, normed);
        ds3_ref_vec_matmul(normed, output_w, ref_logits,
                           DS3_N_EMBD, DS3_N_VOCAB);

        const float *eng_logits = engine_logits + (size_t)step * DS3_N_VOCAB;
        const float *eng_hidden = engine_hidden + (size_t)step * DS3_N_EMBD;

        float max_logit_diff = 0.0f;
        int max_logit_idx = -1;
        for (int i = 0; i < DS3_N_VOCAB; i++) {
            float diff = fabsf(eng_logits[i] - ref_logits[i]);
            if (diff > max_logit_diff) {
                max_logit_diff = diff;
                max_logit_idx = i;
            }
        }

        int engine_top = 0, ref_top = 0;
        for (int i = 1; i < DS3_N_VOCAB; i++) {
            if (eng_logits[i] > eng_logits[engine_top]) engine_top = i;
            if (ref_logits[i] > ref_logits[ref_top]) ref_top = i;
        }

        fprintf(stderr,
                "[decode compare step=%d tok=%d] logit_max_diff=%.6e @ idx=%d "
                "engine=%.6f ref=%.6f top(engine=%d ref=%d)%s\n",
                step, tokens[step], max_logit_diff, max_logit_idx,
                max_logit_idx >= 0 ? eng_logits[max_logit_idx] : 0.0f,
                max_logit_idx >= 0 ? ref_logits[max_logit_idx] : 0.0f,
                engine_top, ref_top,
                engine_top != ref_top ? " TOP_MISMATCH" : "");

        if (max_logit_diff > 1e-3f && first_bad_step < 0) {
            first_bad_step = step;
        }

        float max_hidden_diff = 0.0f;
        int max_hidden_dim = -1;
        for (int d = 0; d < DS3_N_EMBD; d++) {
            float diff = fabsf(eng_hidden[d] - normed[d]);
            if (diff > max_hidden_diff) {
                max_hidden_diff = diff;
                max_hidden_dim = d;
            }
        }
        if (max_hidden_diff > 1e-3f || engine_top != ref_top) {
            fprintf(stderr,
                    "[decode compare step=%d] hidden_max_diff=%.6e @ dim=%d "
                    "engine=%.6f ref=%.6f\n",
                    step, max_hidden_diff, max_hidden_dim,
                    max_hidden_dim >= 0 ? eng_hidden[max_hidden_dim] : 0.0f,
                    max_hidden_dim >= 0 ? normed[max_hidden_dim] : 0.0f);
            if (first_bad_step < 0) first_bad_step = step;
        }
    }

    /* Cleanup. */
    free(normed); free(ref_logits);
    free(layer_out);
    free(engine_hidden); free(engine_logits);
    free(output_norm_w); free(output_w);
    for (int l = 0; l < DS3_N_LAYER; l++) {
        free(kv_cache[l].k_cache); free(kv_cache[l].v_cache);
    }
    free(kv_cache);
    free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
    free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);

    fprintf(stderr, "[decode compare] first_bad_step=%d (n_steps=%d)\n",
            first_bad_step, n_steps);
    return (first_bad_step < 0) ? 0 : 1;
}

int ds3_engine_debug_per_layer_compare(ds3_engine_t *e, int token_id)
{
    if (!e || token_id < 0 || token_id >= DS3_N_VOCAB) return -1;

    float *input = (float *)malloc(DS3_N_EMBD * sizeof(float));
    float *cur = (float *)malloc(DS3_N_EMBD * sizeof(float));
    float *next = (float *)malloc(DS3_N_EMBD * sizeof(float));
    float *engine_hidden = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!input || !cur || !next || !engine_hidden ||
        ds3_ref_dequantize_row(e->weights->token_embd, token_id, input) != 0) {
        free(input); free(cur); free(next); free(engine_hidden);
        return -1;
    }
    memcpy(cur, input, DS3_N_EMBD * sizeof(float));

    ds3_layer_state_t state = {0};
    float *xb          = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *q           = (float *)calloc(DS3_Q_DIM, sizeof(float));
    float *k           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *v           = (float *)calloc(DS3_KV_DIM, sizeof(float));
    float *attn_out    = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *ffn_out     = (float *)calloc(DS3_N_EMBD, sizeof(float));
    float *gate_logits = (float *)calloc(DS3_N_EXPERT, sizeof(float));
    float *exp_scores  = (float *)calloc(DS3_N_EXPERT_USED, sizeof(float));
    int   *exp_indices = (int *)calloc(DS3_N_EXPERT_USED, sizeof(int));
    float *attn_score  = (float *)calloc(DS3_N_HEAD * 1, sizeof(float));

    if (!xb || !q || !k || !v || !attn_out || !ffn_out ||
        !gate_logits || !exp_scores || !exp_indices || !attn_score) {
        free(cur); free(next); free(engine_hidden);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }

    state.xb = xb;
    state.q = q; state.k = k; state.v = v;
    state.attn_out = attn_out; state.ffn_out = ffn_out;
    state.gate_logits = gate_logits;
    state.expert_weights = exp_scores;
    state.expert_indices = exp_indices;
    state.attn_score = attn_score;

    float *postattn_capture = (float *)calloc((size_t)DS3_N_LAYER * DS3_N_EMBD, sizeof(float));
    if (!postattn_capture) {
        free(cur); free(next); free(engine_hidden);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }
    g_debug_postattn_token = postattn_capture;

    float *engine_prev = (float *)malloc(DS3_N_EMBD * sizeof(float));
    if (!engine_prev) {
        g_debug_postattn_token = NULL;
        free(postattn_capture);
        free(cur); free(next); free(engine_hidden);
        free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
        free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);
        return -1;
    }
    memcpy(engine_prev, input, DS3_N_EMBD * sizeof(float));

    int first_bad = -1;
    for (int L = 0; L < DS3_N_LAYER; L++) {
        /* ---- Reference: full layer L ---- */
        ds3_kv_cache_t kv_cache = {0};
        kv_cache.k_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
        kv_cache.v_cache = (float *)calloc(DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
        kv_cache.capacity = 1;
        kv_cache.seq_len  = 0;
        if (!kv_cache.k_cache || !kv_cache.v_cache) {
            free(kv_cache.k_cache); free(kv_cache.v_cache);
            break;
        }
        ds3_ref_layer_forward(&e->weights->layers[L], cur, &kv_cache, 0, 1, next, &state);
        free(kv_cache.k_cache);
        free(kv_cache.v_cache);

        /* Reference raw router logits for layer L (kept for comparisons below). */
        float *ref_logits = NULL;
        float *ffn_norm_w = ds3_ref_dequantize_tensor(e->weights->layers[L].ffn_norm);
        float *gate_inp_w = ds3_ref_dequantize_tensor(e->weights->layers[L].ffn_gate_inp);
        float *ref_normed = (float *)malloc(DS3_N_EMBD * sizeof(float));
        if (ffn_norm_w && gate_inp_w && ref_normed) {
            ref_logits = (float *)malloc(DS3_N_EXPERT * sizeof(float));
            if (ref_logits) {
                ds3_ref_rms_norm(state.attn_out, ffn_norm_w, DS3_NORM_EPS,
                                 DS3_N_EMBD, ref_normed);
                ds3_ref_vec_matmul(ref_normed, gate_inp_w, ref_logits,
                                   DS3_N_EMBD, DS3_N_EXPERT);
            }
        }
        free(ffn_norm_w); free(gate_inp_w); free(ref_normed);

        /* ---- Engine: isolated attention branch for layer L ---- */
        if (ds3_metal_buffer_write(e->buf_hidden, 0, cur,
                                   DS3_N_EMBD * sizeof(float)) != 0) {
            fprintf(stderr, "[per-layer compare] buf_hidden write failed at layer %d\n", L);
            break;
        }
        if (run_attention_single_layer(e, L, 0) != 0) {
            fprintf(stderr, "[per-layer compare] attention single-layer failed at layer %d\n", L);
            break;
        }
        float *engine_attn = (float *)malloc(DS3_N_EMBD * sizeof(float));
        if (!engine_attn) break;
        if (ds3_metal_buffer_read(e->buf_hidden, 0, engine_attn,
                                  DS3_N_EMBD * sizeof(float)) != 0) {
            free(engine_attn);
            break;
        }

        float max_diff_attn = 0.0f;
        for (int d = 0; d < DS3_N_EMBD; d++) {
            float diff = fabsf(engine_attn[d] - state.attn_out[d]);
            if (diff > max_diff_attn) max_diff_attn = diff;
        }

        /* ---- Engine: isolated router matmul for layer L ---- */
        ds3_metal_buffer_t *logit_scratch = ds3_metal_buffer_alloc(
            DS3_N_EXPERT * sizeof(float));
        float *engine_router_logits2 = (float *)malloc(DS3_N_EXPERT * sizeof(float));
        float router_logit_diff2 = 0.0f;
        if (logit_scratch && engine_router_logits2 &&
            ds3_metal_buffer_write(e->buf_hidden, 0, engine_attn,
                                   DS3_N_EMBD * sizeof(float)) == 0) {
            ds3_layer_weights_t *lw = &e->weights->layers[L];
            ds3_metal_begin_batch();
            ds3_metal_rms_norm(e->buf_hidden, e->layer_bufs[L].ffn_norm,
                               e->buf_norm, DS3_N_EMBD, 1, DS3_NORM_EPS);
            ds3_metal_moe_ffn_router(
                e->buf_norm, e->layer_bufs[L].ffn_gate_inp, logit_scratch,
                DS3_N_EMBD, DS3_N_EXPERT,
                lw->ffn_gate_inp->nb[1],
                lw->ffn_gate_inp->type);
            ds3_metal_end_batch();
            if (ds3_metal_buffer_read(logit_scratch, 0, engine_router_logits2,
                                      DS3_N_EXPERT * sizeof(float)) == 0 &&
                ref_logits) {
                for (int i = 0; i < DS3_N_EXPERT; i++) {
                    float d = fabsf(engine_router_logits2[i] - ref_logits[i]);
                    if (d > router_logit_diff2) router_logit_diff2 = d;
                }
            }
        }
        ds3_metal_buffer_free(logit_scratch);
        free(engine_router_logits2);
        /* ref_logits is kept for the full-engine logit comparison below. */

        /* ---- Engine: isolated FFN branch for layer L ---- */
        float *engine_ffn = (float *)malloc(DS3_N_EMBD * sizeof(float));
        if (!engine_ffn) break;
        if (run_ffn_single_layer(e, L, engine_attn,
                                 state.expert_indices,
                                 state.expert_weights) != 0 ||
            ds3_metal_buffer_read(e->buf_ffn_out, 0, engine_ffn,
                                  DS3_N_EMBD * sizeof(float)) != 0) {
            free(engine_ffn);
            break;
        }

        float max_diff_ffn_iso = 0.0f;
        for (int d = 0; d < DS3_N_EMBD; d++) {
            float diff = fabsf(engine_ffn[d] - state.ffn_out[d]);
            if (diff > max_diff_ffn_iso) max_diff_ffn_iso = diff;
        }
        free(engine_ffn);

        /* ---- Engine: full layers 0..L ---- */
        e->seq_len = 0;
        g_debug_stop_full_layer = L;
        int rc = forward_token(e, token_id, 0.0f);
        g_debug_stop_full_layer = -1;
        if (rc != 0) {
            fprintf(stderr, "[per-layer compare] forward_token failed at layer %d\n", L);
            break;
        }
        if (ds3_metal_buffer_read(e->buf_hidden, 0, engine_hidden,
                                  DS3_N_EMBD * sizeof(float)) != 0) {
            fprintf(stderr, "[per-layer compare] buffer read failed at layer %d\n", L);
            break;
        }

        float max_diff_full = 0.0f;
        int max_d_full = -1;
        for (int d = 0; d < DS3_N_EMBD; d++) {
            float diff = fabsf(engine_hidden[d] - next[d]);
            if (diff > max_diff_full) {
                max_diff_full = diff;
                max_d_full = d;
            }
        }

        float max_diff_ffn = 0.0f;
        for (int d = 0; d < DS3_N_EMBD; d++) {
            float engine_ffn = engine_hidden[d] - engine_attn[d];
            float ref_ffn    = next[d] - state.attn_out[d];
            float diff = fabsf(engine_ffn - ref_ffn);
            if (diff > max_diff_ffn) max_diff_ffn = diff;
        }

        float input_diff = 0.0f;
        for (int d = 0; d < DS3_N_EMBD; d++) {
            float diff = fabsf(engine_prev[d] - cur[d]);
            if (diff > input_diff) input_diff = diff;
        }

        fprintf(stderr,
                "[per-layer compare] layer=%02d input_diff=%.6e attn_diff=%.6e "
                "ffn_iso_diff=%.6e ffn_diff=%.6e full_diff=%.6e @ dim=%d "
                "engine=%.6f ref=%.6f\n",
                L, input_diff, max_diff_attn, max_diff_ffn_iso, max_diff_ffn, max_diff_full, max_d_full,
                max_d_full >= 0 ? engine_hidden[max_d_full] : 0.0f,
                max_d_full >= 0 ? next[max_d_full] : 0.0f);

        free(engine_attn);

        /* Compare full-engine router logits for this layer. */
        float logit_diff = 0.0f;
        if (ref_logits) {
            float *engine_logits = (float *)malloc(DS3_N_EXPERT * sizeof(float));
            if (engine_logits &&
                ds3_metal_buffer_read(e->buf_gate_logits,
                                      (size_t)L * DS3_N_EXPERT * sizeof(float),
                                      engine_logits,
                                      DS3_N_EXPERT * sizeof(float)) == 0) {
                for (int i = 0; i < DS3_N_EXPERT; i++) {
                    float d = fabsf(engine_logits[i] - ref_logits[i]);
                    if (d > logit_diff) logit_diff = d;
                }
            }
            free(engine_logits);
        }

        /* Compare router top-k indices/scores for this layer. */
        int32_t engine_idx[DS3_N_EXPERT_USED];
        float   engine_score[DS3_N_EXPERT_USED];
        int idx_mismatch = 0;
        float score_diff = 0.0f;
        if (ds3_metal_buffer_read(e->buf_router_indices,
                                  (size_t)L * DS3_N_EXPERT_USED * sizeof(int32_t),
                                  engine_idx,
                                  DS3_N_EXPERT_USED * sizeof(int32_t)) == 0 &&
            ds3_metal_buffer_read(e->buf_router_scores,
                                  (size_t)L * DS3_N_EXPERT_USED * sizeof(float),
                                  engine_score,
                                  DS3_N_EXPERT_USED * sizeof(float)) == 0) {
            for (int i = 0; i < DS3_N_EXPERT_USED; i++) {
                if ((int)engine_idx[i] != state.expert_indices[i]) idx_mismatch++;
                float d = fabsf(engine_score[i] - state.expert_weights[i]);
                if (d > score_diff) score_diff = d;
            }
        }
        fprintf(stderr,
                "[per-layer compare] layer=%02d router_logit_iso=%.6e "
                "router_logit_full=%.6e router_idx_mismatch=%d/%d "
                "router_score_maxdiff=%.6e\n",
                L, router_logit_diff2, logit_diff, idx_mismatch,
                DS3_N_EXPERT_USED, score_diff);

        free(ref_logits);

        if (max_diff_full > 1e-3f && first_bad < 0) first_bad = L;

        memcpy(engine_prev, engine_hidden, DS3_N_EMBD * sizeof(float));
        float *tmp = cur; cur = next; next = tmp;
    }

    free(engine_prev);

    g_debug_postattn_token = NULL;
    free(postattn_capture);

    free(input); free(cur); free(next); free(engine_hidden);
    free(xb); free(q); free(k); free(v); free(attn_out); free(ffn_out);
    free(gate_logits); free(exp_scores); free(exp_indices); free(attn_score);

    return (first_bad < 0) ? 0 : 1;
}

int ds3_engine_generate(ds3_engine_t *e,
                        const int *prompt_tokens,
                        int n_prompt,
                        int n_predict,
                        float temperature,
                        int *output_tokens,
                        int output_capacity)
{
    return ds3_engine_generate_ex(e, prompt_tokens, n_prompt, n_predict,
                                  temperature, output_tokens, output_capacity,
                                  NULL, NULL);
}

/* ============================================================================
 * KV-cache provider helpers
 * ============================================================================ */

static size_t kv_cache_layer_bytes(int n_tokens)
{
    return (size_t)n_tokens * DS3_N_HEAD_KV * DS3_HEAD_DIM * sizeof(uint16_t);
}

static int ensure_kv_cache_host(ds3_engine_t *e)
{
    if (e->kv_cache_host) return 0;
    size_t bytes = (size_t)DS3_N_LAYER * 2 * kv_cache_layer_bytes(e->n_ctx);
    e->kv_cache_host = calloc(1, bytes);
    if (!e->kv_cache_host) {
        fprintf(stderr, "[engine] failed to allocate host KV cache (%zu bytes)\n", bytes);
        return -1;
    }
    e->kv_cache_host_bytes = bytes;
    return 0;
}

static void free_kv_cache_host(ds3_engine_t *e)
{
    free(e->kv_cache_host);
    e->kv_cache_host = NULL;
    e->kv_cache_host_bytes = 0;
}

/* Download n_tokens of KV data from the GPU cache into a contiguous host
 * buffer laid out as [L0 K][L0 V][L1 K][L1 V]... */
static int kv_cache_download(ds3_engine_t *e, int n_tokens, void *host)
{
    if (!host || n_tokens <= 0 || n_tokens > e->n_ctx) return -1;
    size_t layer_bytes = kv_cache_layer_bytes(n_tokens);
    uint8_t *p = (uint8_t *)host;
    for (int l = 0; l < DS3_N_LAYER; l++) {
        if (ds3_metal_buffer_read(e->kv_k[l], 0, p, layer_bytes) != 0) return -1;
        p += layer_bytes;
        if (ds3_metal_buffer_read(e->kv_v[l], 0, p, layer_bytes) != 0) return -1;
        p += layer_bytes;
    }
    return 0;
}

/* Upload n_tokens of KV data from a contiguous host buffer into the GPU
 * cache at offset 0. */
static int kv_cache_upload(ds3_engine_t *e, int n_tokens, const void *host)
{
    if (!host || n_tokens <= 0 || n_tokens > e->n_ctx) return -1;
    size_t layer_bytes = kv_cache_layer_bytes(n_tokens);
    const uint8_t *p = (const uint8_t *)host;
    for (int l = 0; l < DS3_N_LAYER; l++) {
        if (ds3_metal_buffer_write(e->kv_k[l], 0, p, layer_bytes) != 0) return -1;
        p += layer_bytes;
        if (ds3_metal_buffer_write(e->kv_v[l], 0, p, layer_bytes) != 0) return -1;
        p += layer_bytes;
    }
    return 0;
}

void ds3_engine_set_kv_provider(ds3_engine_t *e, ds3_kv_cache_provider_t *provider)
{
    if (!e) return;
    e->kv_provider = provider;
}

void ds3_engine_set_session_id(ds3_engine_t *e, const char *session_id)
{
    if (!e) return;
    if (!session_id || !session_id[0]) {
        e->session_id[0] = '\0';
        return;
    }
    strncpy(e->session_id, session_id, sizeof(e->session_id) - 1);
    e->session_id[sizeof(e->session_id) - 1] = '\0';
}

static bool is_stop_token(const ds3_engine_t *e, int token_id)
{
    if (token_id < 0) return false;
    if (token_id == e->vocab.eos_id) return true;
    if (token_id == e->vocab.im_end_id) return true;
    /* Temporarily disabled: stopping on </think> truncates the final answer
     * for the Qwen3 GGUF we are testing.  Let the token be emitted and strip
     * the think block in the caller. */
    /* if (token_id == e->vocab.think_end_id) return true; */
    return false;
}

int ds3_engine_generate_ex(ds3_engine_t *e,
                           const int *prompt_tokens,
                           int n_prompt,
                           int n_predict,
                           float temperature,
                           int *output_tokens,
                           int output_capacity,
                           ds3_engine_logit_cb_t cb,
                           void *cb_user)
{
    if (!e || !prompt_tokens || n_prompt <= 0 || !output_tokens || output_capacity <= 0) {
        return -1;
    }
    if (n_prompt > e->n_ctx) {
        fprintf(stderr, "[engine] prompt length %d exceeds n_ctx %d\n",
                n_prompt, e->n_ctx);
        return -1;
    }

    srand((unsigned)time(NULL));
    e->seq_len = 0;

    if (getenv("DS3_CHUNK_DEBUG") != NULL) {
        debug_compare_chunk_and_token(e, prompt_tokens, n_prompt);
    }

    ds3_metal_profile_reset();
    int next_token = -1;

    /* -------------------------------------------------------------------------
     * KV-cache prefix lookup
     * ------------------------------------------------------------------------- */
    int cached_len = 0;
    if (e->kv_provider && e->session_id[0] != '\0') {
        if (e->kv_provider->vtable->lookup(e->kv_provider, e->session_id,
                                           prompt_tokens, n_prompt,
                                           &cached_len) == 0) {
            if (cached_len < 0) cached_len = 0;
            if (cached_len > n_prompt) cached_len = n_prompt;
            if (cached_len > 0) {
                if (ensure_kv_cache_host(e) == 0 &&
                    e->kv_provider->vtable->read(e->kv_provider, e->session_id,
                                                 0, cached_len, e->kv_cache_host) == 0 &&
                    kv_cache_upload(e, cached_len, e->kv_cache_host) == 0) {
                    e->seq_len = cached_len;
                    ds3_log_info("[engine] KV cache hit: %d / %d tokens\n",
                                 cached_len, n_prompt);
                } else {
                    cached_len = 0;
                    e->seq_len = 0;
                }
            }
        } else {
            cached_len = 0;
        }
    }

    /* -------------------------------------------------------------------------
     * Prefill remaining tokens
     * ------------------------------------------------------------------------- */
    if (cached_len == n_prompt && n_prompt > 0) {
        /* Entire prompt is cached.  Run the last token once to obtain the
         * first decode logit. */
        e->seq_len = n_prompt - 1;
        next_token = forward_token(e, prompt_tokens[n_prompt - 1], temperature);
        if (next_token < 0) return -1;
        e->seq_len++;
    } else {
        /* Chunk prefill is now the default.  Users can opt out with
         * DS3_NO_CHUNK_PREFILL=1 or (legacy) DS3_USE_CHUNK_PREFILL=0.  Very short
         * prompts fall back to the token path because the batched dispatch overhead
         * is not worth it for just a few tokens. */
        bool use_chunk_prefill = true;
        const char *no_chunk_env = getenv("DS3_NO_CHUNK_PREFILL");
        const char *use_chunk_env = getenv("DS3_USE_CHUNK_PREFILL");
        if (no_chunk_env != NULL) {
            use_chunk_prefill = false;
        } else if (use_chunk_env != NULL &&
                   (strcmp(use_chunk_env, "0") == 0 ||
                    strcmp(use_chunk_env, "false") == 0 ||
                    strcmp(use_chunk_env, "no") == 0)) {
            use_chunk_prefill = false;
        }
        if (n_prompt - cached_len < DS3_PREFILL_FALLBACK_TOKENS) {
            use_chunk_prefill = false;
        }

        if (use_chunk_prefill) {
            int chunk_size = DS3_PREFILL_CHUNK_SIZE;
            const char *chunk_env = getenv("DS3_PREFILL_CHUNK_SIZE");
            if (chunk_env) {
                int v = atoi(chunk_env);
                if (v > 0) chunk_size = v;
            }

            int prompt_pos = cached_len;
            while (prompt_pos < n_prompt) {
                int remaining = n_prompt - prompt_pos;
                int chunk = remaining < chunk_size ? remaining : chunk_size;
                next_token = forward_chunk(e, prompt_tokens + prompt_pos, chunk, temperature);
                if (next_token < 0) return -1;
                e->seq_len += chunk;
                prompt_pos += chunk;
            }
        } else {
            for (int i = cached_len; i < n_prompt; i++) {
                next_token = forward_token(e, prompt_tokens[i], temperature);
                if (next_token < 0) return -1;
                e->seq_len++;
            }
        }
    }
    ds3_metal_profile_print("prompt");

    int n_gen = 0;
    for (int i = 0; i < n_predict && n_gen < output_capacity && e->seq_len < e->n_ctx; i++) {
        if (is_stop_token(e, next_token)) break;

        output_tokens[n_gen++] = next_token;
        if (cb) cb(i, e->logits_host, DS3_N_VOCAB, next_token, cb_user);

        ds3_metal_profile_reset();
        next_token = forward_token(e, next_token, temperature);
        if (next_token < 0) break;
        e->seq_len++;

        char label[32];
        snprintf(label, sizeof(label), "step=%d", i);
        ds3_metal_profile_print(label);
    }

    /* Write prompt + generated tokens KV back so future turns can reuse
     * the full prefix including the assistant response. */
    int total_len = n_prompt + n_gen;
    if (e->kv_provider && e->session_id[0] != '\0' && total_len > 0) {
        if (ensure_kv_cache_host(e) == 0 &&
            kv_cache_download(e, total_len, e->kv_cache_host) == 0) {
            size_t kv_bytes = (size_t)DS3_N_LAYER * 2 * kv_cache_layer_bytes(total_len);

            /* Build merged token array: prompt + generated */
            int *all_tokens = (int *)malloc((size_t)total_len * sizeof(int));
            if (all_tokens) {
                memcpy(all_tokens, prompt_tokens, (size_t)n_prompt * sizeof(int));
                memcpy(all_tokens + n_prompt, output_tokens, (size_t)n_gen * sizeof(int));
                int write_rc = e->kv_provider->vtable->write(e->kv_provider, e->session_id,
                                                             all_tokens, total_len,
                                                             e->kv_cache_host, kv_bytes);
                free(all_tokens);
                if (write_rc != 0) {
                    ds3_log_warn("[engine] KV cache write failed for session %s (rc=%d)\n",
                                 e->session_id, write_rc);
                }
            }
        }
    }

    return n_gen;
}

/* ============================================================================
 * Public utilities
 * ============================================================================ */

int ds3_tokenize(const struct ds3_engine *e, const char *text,
                 int *tokens, int max_tokens)
{
    if (!e || !text || !tokens || max_tokens <= 0) return -1;
    return ds3_vocab_encode(&e->vocab, text, tokens, max_tokens);
}

void ds3_token_to_str(const struct ds3_engine *e, int token_id,
                      char *buf, size_t buf_size)
{
    if (!e || !buf || buf_size == 0) return;
    buf[0] = '\0';
    if (token_id < 0 || token_id >= e->vocab.n_vocab) return;
    const char *raw = ds3_vocab_decode(&e->vocab, token_id);
    if (!raw) return;
    strncpy(buf, raw, buf_size - 1);
    buf[buf_size - 1] = '\0';
}

void ds3_print_model_info(const ds3_engine_t *e)
{
    if (!e) return;
    ds3_weights_print_summary(e->weights);
    ds3_print_info("\nKV cache: %d layers x %d tokens x %d kv_heads x %d head_dim (FP16)\n",
           DS3_N_LAYER, e->n_ctx, DS3_N_HEAD_KV, DS3_HEAD_DIM);
}

int ds3_engine_chat_format(const ds3_engine_t *e, const char *system,
                           const char *user, int *tokens, int max_tokens)
{
    if (!e || !user || !tokens || max_tokens <= 0) return -1;
    return ds3_chat_format(&e->vocab, system, user, tokens, max_tokens);
}

int ds3_engine_decode_sequence(const ds3_engine_t *e,
                               const int *tokens, int n_tokens,
                               char *buf, size_t buf_size)
{
    if (!e || !tokens || n_tokens <= 0 || !buf || buf_size == 0) return -1;
    return ds3_vocab_decode_sequence(&e->vocab, tokens, n_tokens, buf, buf_size);
}
