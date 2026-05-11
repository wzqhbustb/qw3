/*
 * ds3_reference.c — CPU FP32 Reference Implementation
 *
 * Single-threaded, unoptimized, mathematically correct.
 * Intended as ground-truth for Metal kernel validation.
 */

#include "ds3_reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ============================================================================
 * 1. Dequantization — extracted from llama.cpp ggml-quants.c
 * ============================================================================ */

/* FP16 ↔ FP32 (IEEE 754 half-precision).
 * Use native _Float16 when available (clang/GCC on Apple Silicon),
 * fall back to software decode for MSVC and other compilers. */
#ifdef __FLT16_MAX__
static inline float fp16_to_fp32(uint16_t h)
{
    _Float16 f;
    memcpy(&f, &h, sizeof(f));
    return (float)f;
}
#else
static inline float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t u = sign;
            float f; memcpy(&f, &u, sizeof(f)); return f;
        }
        uint32_t e = 1;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            e--;
        }
        mant &= 0x3FF;
        exp = e + 112;
    } else if (exp == 0x1F) {
        exp = 0xFF;
    } else {
        exp = exp + 112;
    }

    uint32_t u = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &u, sizeof(f)); return f;
}
#endif

/* -------------------------------------------------------------------------- */
/* Q4_K (Q4_K_M) — 256 weights / super-block, 4.5 bits per weight            */
/* -------------------------------------------------------------------------- */

#define DS3_Q4K_BLOCK_SIZE  256
#define DS3_Q4K_SCALE_SIZE  12

typedef struct {
    uint16_t d;                 /* super-block scale (FP16) */
    uint16_t dmin;              /* super-block min scale (FP16) */
    uint8_t  scales[DS3_Q4K_SCALE_SIZE]; /* 6-bit quantized scales & mins */
    uint8_t  qs[DS3_Q4K_BLOCK_SIZE / 2]; /* 4-bit quantized weights */
} ds3_block_q4_K;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(ds3_block_q4_K) == 144, "ds3_block_q4_K size mismatch with GGUF");
#endif

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequantize_row_q4_K(const ds3_block_q4_K *x, float *y, int64_t k)
{
    const int nb = (int)(k / DS3_Q4K_BLOCK_SIZE);

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < DS3_Q4K_BLOCK_SIZE; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >>   4) - m2;

            q += 32;
            is += 2;
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Q6_K — 256 weights / super-block, 6.5625 bits per weight                   */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint8_t  ql[128];   /* quants, lower 4 bits */
    uint8_t  qh[64];    /* quants, upper 2 bits */
    int8_t   scales[16];/* scales per 16-weight group */
    uint16_t d;         /* super-block scale (FP16) */
} ds3_block_q6_K;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(ds3_block_q6_K) == 210, "ds3_block_q6_K size mismatch with GGUF");
#endif

static void dequantize_row_q6_K(const ds3_block_q6_K *x, float *y, int64_t k)
{
    const int nb = (int)(k / DS3_Q4K_BLOCK_SIZE);

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        for (int n = 0; n < DS3_Q4K_BLOCK_SIZE; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Q8_0 — 32 weights / block, 8.5 bits per weight                             */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint16_t d;       /* delta (FP16) */
    int8_t   qs[32];  /* quants */
} ds3_block_q8_0;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(ds3_block_q8_0) == 34, "ds3_block_q8_0 size mismatch with GGUF");
#endif

static void dequantize_row_q8_0(const ds3_block_q8_0 *x, float *y, int64_t k)
{
    const int qk = 32;
    const int nb = (int)(k / qk);

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i * qk + j] = x[i].qs[j] * d;
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Main entry point                                                           */
/* -------------------------------------------------------------------------- */

/* ============================================================================
 * 1a. Row dequantization (for token embedding lookup)
 * ============================================================================ */

int ds3_ref_dequantize_row(const ds3_tensor_t *t, int row, float *out)
{
    if (!t || !t->data || !out) return -1;
    if (t->n_dims < 2) return -1;

    const int64_t n_rows = (int64_t)t->ne[1];
    const int64_t n_cols = (int64_t)t->ne[0];
    if (row < 0 || row >= n_rows) {
        fprintf(stderr, "[dequantize_row] row %d out of range [0, %lld)\n",
                row, (long long)n_rows);
        return -1;
    }

    const uint8_t *row_ptr = (const uint8_t *)t->data + row * t->nb[1];

    switch (t->type) {
        case DS3_TYPE_F32: {
            memcpy(out, row_ptr, (size_t)n_cols * sizeof(float));
            return 0;
        }
        case DS3_TYPE_F16: {
            const uint16_t *src = (const uint16_t *)row_ptr;
            for (int64_t i = 0; i < n_cols; i++) {
                out[i] = fp16_to_fp32(src[i]);
            }
            return 0;
        }
        case DS3_TYPE_Q4_K: {
            if (n_cols % DS3_Q4K_BLOCK_SIZE != 0) return -1;
            dequantize_row_q4_K((const ds3_block_q4_K *)row_ptr, out, n_cols);
            return 0;
        }
        case DS3_TYPE_Q6_K: {
            if (n_cols % DS3_Q4K_BLOCK_SIZE != 0) return -1;
            dequantize_row_q6_K((const ds3_block_q6_K *)row_ptr, out, n_cols);
            return 0;
        }
        case DS3_TYPE_Q8_0: {
            const int qk = 32;
            if (n_cols % qk != 0) return -1;
            dequantize_row_q8_0((const ds3_block_q8_0 *)row_ptr, out, n_cols);
            return 0;
        }
        default:
            fprintf(stderr, "[dequantize_row] unsupported type %d\n", t->type);
            return -1;
    }
}

float *ds3_ref_dequantize_tensor(const ds3_tensor_t *t)
{
    if (!t || !t->data) return NULL;

    /* Compute total number of elements */
    int64_t n_elements = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) {
        n_elements *= (int64_t)t->ne[d];
    }

    if (n_elements <= 0) {
        fprintf(stderr, "[dequantize] n_elements=%lld <= 0\n", (long long)n_elements);
        return NULL;
    }

    float *out = (float *)malloc((size_t)n_elements * sizeof(float));
    if (!out) return NULL;

    switch (t->type) {
        case DS3_TYPE_F32: {
            const float *src = (const float *)t->data;
            memcpy(out, src, (size_t)n_elements * sizeof(float));
            break;
        }
        case DS3_TYPE_F16: {
            const uint16_t *src = (const uint16_t *)t->data;
            for (int64_t i = 0; i < n_elements; i++) {
                out[i] = fp16_to_fp32(src[i]);
            }
            break;
        }
        case DS3_TYPE_Q4_K: {
            if (n_elements % DS3_Q4K_BLOCK_SIZE != 0) {
                free(out);
                return NULL;
            }
            size_t expected_size = (size_t)(n_elements / DS3_Q4K_BLOCK_SIZE) * sizeof(ds3_block_q4_K);
            if (t->size < expected_size) {
                free(out);
                return NULL;
            }
            const ds3_block_q4_K *src = (const ds3_block_q4_K *)t->data;
            dequantize_row_q4_K(src, out, n_elements);
            break;
        }
        case DS3_TYPE_Q6_K: {
            if (n_elements % DS3_Q4K_BLOCK_SIZE != 0) {
                free(out);
                return NULL;
            }
            size_t expected_size = (size_t)(n_elements / DS3_Q4K_BLOCK_SIZE) * sizeof(ds3_block_q6_K);
            if (t->size < expected_size) {
                free(out);
                return NULL;
            }
            const ds3_block_q6_K *src = (const ds3_block_q6_K *)t->data;
            dequantize_row_q6_K(src, out, n_elements);
            break;
        }
        case DS3_TYPE_Q8_0: {
            const int qk = 32;
            if (n_elements % qk != 0) {
                free(out);
                return NULL;
            }
            size_t expected_size = (size_t)(n_elements / qk) * sizeof(ds3_block_q8_0);
            if (t->size < expected_size) {
                free(out);
                return NULL;
            }
            const ds3_block_q8_0 *src = (const ds3_block_q8_0 *)t->data;
            dequantize_row_q8_0(src, out, n_elements);
            break;
        }
        default:
            free(out);
            return NULL;
    }

    return out;
}

/* ============================================================================
 * 2. Basic Operators
 * ============================================================================ */

void ds3_ref_rms_norm(const float *x, const float *weight, float eps,
                      int n, float *out)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss /= n;
    ss += eps;
    float rms = 1.0f / sqrtf(ss);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

void ds3_ref_softmax(float *x, int n)
{
    float maxv = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > maxv) maxv = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void ds3_ref_rope(float *q, float *k,
                  int seq_pos,
                  int n_q_head, int n_kv_head, int head_dim)
{
    /* GPT-NeoX style RoPE for Qwen3.
     * The rotation pairs are (d, d + head_dim/2), not adjacent (d, d+1).
     *   theta_i = theta_base ^ (-2i / head_dim)
     * This matches llama.cpp's LLAMA_ROPE_TYPE_NEOX for the qwen3moe arch. */
    const float theta = DS3_ROPE_THETA;
    const float theta_scale = -2.0f / head_dim;
    const int n_pairs = head_dim / 2;

    /* Q heads */
    for (int h = 0; h < n_q_head; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < n_pairs; i++) {
            float freq = powf(theta, (float)i * theta_scale);
            float angle = seq_pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float v0 = qh[i];
            float v1 = qh[i + n_pairs];
            qh[i]           = v0 * c - v1 * s;
            qh[i + n_pairs] = v0 * s + v1 * c;
        }
    }

    /* KV heads */
    for (int h = 0; h < n_kv_head; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < n_pairs; i++) {
            float freq = powf(theta, (float)i * theta_scale);
            float angle = seq_pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float v0 = kh[i];
            float v1 = kh[i + n_pairs];
            kh[i]           = v0 * c - v1 * s;
            kh[i + n_pairs] = v0 * s + v1 * c;
        }
    }
}

void ds3_ref_matmul(const float *A, const float *B, float *C,
                    int M, int N, int K)
{
    /* C[M][N] = A[M][K] × B[K][N]
     * All row-major. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* Vector×matrix: y = x @ W^T  (or y[j] = sum_i x[i] * W[j][i])
 * x: [in_dim], W: [out_dim][in_dim] row-major, y: [out_dim]
 * GGUF stores weights as [out_dim, in_dim] row-major.
 * Used for all linear projections (Q, K, V, O, router, etc.) */
void ds3_ref_vec_matmul(const float *x, const float *W, float *y,
                        int in_dim, int out_dim)
{
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * W[j * in_dim + i];
        }
        y[j] = sum;
    }
}

void ds3_ref_silu(const float *x, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void ds3_ref_mul(const float *a, const float *b, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void ds3_ref_copy(const float *in_, float *out, int n)
{
    memcpy(out, in_, n * sizeof(float));
}

void ds3_ref_add(const float *a, const float *b, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

/* ============================================================================
 * 3. Top-k (naive sort for N_EXPERT=128)
 * ============================================================================ */

void ds3_ref_topk(const float *scores, int n, int k,
                  int *indices_out, float *values_out)
{
    /* Simple selection sort for small n (128).
     * Pick the largest k elements. */
    int *used = calloc((size_t)n, sizeof(int));
    if (!used) {
        fprintf(stderr, "[topk] calloc failed (n=%d)\n", n);
        return;
    }
    for (int i = 0; i < k; i++) {
        float best = -1e30f;
        int best_idx = -1;
        for (int j = 0; j < n; j++) {
            if (!used[j] && scores[j] > best) {
                best = scores[j];
                best_idx = j;
            }
        }
        used[best_idx] = 1;
        indices_out[i] = best_idx;
        values_out[i] = best;
    }
    free(used);
}

/* ============================================================================
 * 4. Attention
 * ============================================================================ */

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
    float *attn_scores)
{
    /* For Phase 1.4 single-token decode: n_tokens = 1 */
    /* For prefill: n_tokens > 1 (future Phase 2) */
    (void)n_tokens; /* decode-only for now */

    const int n_q_head  = DS3_N_HEAD;
    const int n_kv_head = DS3_N_HEAD_KV;
    const int head_dim  = DS3_HEAD_DIM;
    const int q_dim     = DS3_Q_DIM;
    const int kv_dim    = DS3_KV_DIM;
    const int n_embd    = DS3_N_EMBD;
    const float scale   = 1.0f / sqrtf((float)head_dim);

    /* --- 1. Q/K/V projections --- */
    ds3_ref_vec_matmul(input, w_q, q, n_embd, q_dim);
    ds3_ref_vec_matmul(input, w_k, k, n_embd, kv_dim);
    ds3_ref_vec_matmul(input, w_v, v, n_embd, kv_dim);

    /* --- 2. QK Norm (if present) --- */
#if DS3_HAS_QK_NORM
    for (int h = 0; h < n_q_head; h++) {
        ds3_ref_rms_norm(q + h * head_dim, w_q_norm, DS3_NORM_EPS, head_dim, q + h * head_dim);
    }
    for (int h = 0; h < n_kv_head; h++) {
        ds3_ref_rms_norm(k + h * head_dim, w_k_norm, DS3_NORM_EPS, head_dim, k + h * head_dim);
    }
#endif

    /* --- 3. RoPE --- */
    ds3_ref_rope(q, k, seq_pos, n_q_head, n_kv_head, head_dim);

    /* --- 4. Write K/V to cache --- */
    for (int h = 0; h < n_kv_head; h++) {
        for (int d = 0; d < head_dim; d++) {
            kv_cache->k_cache[seq_pos * kv_dim + h * head_dim + d] = k[h * head_dim + d];
            kv_cache->v_cache[seq_pos * kv_dim + h * head_dim + d] = v[h * head_dim + d];
        }
    }
    /* Update sequence length to include the newly-written token */
    kv_cache->seq_len = seq_pos + 1;

    /* --- 5. Attention scores + softmax + weighted sum --- */
    /* For decode (seq_pos is the newest token):
     *   Compute Q @ K_cache^T for ALL cached positions 0..seq_pos
     *   Softmax over seq_len = seq_pos + 1
     *   Weighted sum of V_cache
     *
     * For prefill: would tile / use FlashAttention (future).
     */
    int total_len = (int)kv_cache->seq_len; /* positions 0..seq_pos (inclusive) */

    /* attention output accumulator: [n_q_head][head_dim] */
    float *attn_out = calloc((size_t)q_dim, sizeof(float));
    if (!attn_out) {
        fprintf(stderr, "[gqa_attention] calloc failed for attn_out (%d floats)\n", q_dim);
        return;
    }

    for (int qh = 0; qh < n_q_head; qh++) {
        /* Each Q head maps to a KV head */
        int kvh = qh / DS3_N_HEAD_PER_KV;

        float *q_head = q + qh * head_dim;

        /* Compute scores: Q @ K^T for all positions */
        for (int pos = 0; pos < total_len; pos++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float k_val = kv_cache->k_cache[pos * kv_dim + kvh * head_dim + d];
                score += q_head[d] * k_val;
            }
            attn_scores[qh * total_len + pos] = score * scale;
        }

        /* Softmax over positions */
        ds3_ref_softmax(attn_scores + qh * total_len, total_len);

        /* Weighted sum of V */
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int pos = 0; pos < total_len; pos++) {
                float v_val = kv_cache->v_cache[pos * kv_dim + kvh * head_dim + d];
                sum += attn_scores[qh * total_len + pos] * v_val;
            }
            attn_out[qh * head_dim + d] = sum;
        }
    }

    /* --- 6. Output projection --- */
    ds3_ref_vec_matmul(attn_out, w_o, output, q_dim, n_embd);

    free(attn_out);
}

/* ============================================================================
 * 5. MoE FFN
 * ============================================================================ */

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
    bool norm_topk_prob)
{

    /* --- 1. Router --- */
    ds3_ref_vec_matmul(input, w_gate_inp, gate_logits, n_embd, n_expert);
    ds3_ref_softmax(gate_logits, n_expert);
    ds3_ref_topk(gate_logits, n_expert, n_used, expert_indices, expert_scores);

    /* --- 2. Norm top-k prob --- */
    if (norm_topk_prob) {
        float sum = 0.0f;
        for (int i = 0; i < n_used; i++) sum += expert_scores[i];
        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            for (int i = 0; i < n_used; i++) expert_scores[i] *= inv_sum;
        }
    }

    /* --- 3. Routed experts (SwiGLU) --- */
    memset(output, 0, n_embd * sizeof(float));

    /* Allocate scratch inside function (reference path — correctness over perf) */
    float *gate_act = calloc((size_t)n_ff_exp, sizeof(float));
    float *up_act   = calloc((size_t)n_ff_exp, sizeof(float));
    float *exp_out  = calloc((size_t)n_embd,  sizeof(float));
    if (!gate_act || !up_act || !exp_out) {
        fprintf(stderr, "[moe_ffn] calloc failed for scratch buffers\n");
        free(gate_act); free(up_act); free(exp_out);
        return;
    }

    for (int i = 0; i < n_used; i++) {
        int eid = expert_indices[i];
        const float *w_g = w_gate_exps + eid * n_ff_exp * n_embd;
        const float *w_u = w_up_exps   + eid * n_ff_exp * n_embd;
        const float *w_d = w_down_exps + eid * n_embd * n_ff_exp;

        /* gate = input @ W_gate */
        ds3_ref_vec_matmul(input, w_g, gate_act, n_embd, n_ff_exp);
        /* silu(gate) */
        ds3_ref_silu(gate_act, gate_act, n_ff_exp);

        /* up = input @ W_up */
        ds3_ref_vec_matmul(input, w_u, up_act, n_embd, n_ff_exp);

        /* silu(gate) * up */
        ds3_ref_mul(gate_act, up_act, gate_act, n_ff_exp);

        /* out = (gate * up) @ W_down */
        ds3_ref_vec_matmul(gate_act, w_d, exp_out, n_ff_exp, n_embd);

        /* weighted accumulate */
        float weight = expert_scores[i];
        for (int j = 0; j < n_embd; j++) {
            output[j] += weight * exp_out[j];
        }
    }

    free(gate_act);
    free(up_act);
    free(exp_out);

    /* --- 4. Shared expert (optional) --- */
    if (w_shared_gate && w_shared_up && w_shared_down) {
        float *sg = calloc((size_t)n_ff_shared, sizeof(float));
        float *su = calloc((size_t)n_ff_shared, sizeof(float));
        float *so = calloc((size_t)n_embd,     sizeof(float));
        if (!sg || !su || !so) {
            fprintf(stderr, "[moe_ffn] calloc failed for shared expert buffers\n");
            free(sg); free(su); free(so);
            return;
        }

        ds3_ref_vec_matmul(input, w_shared_gate, sg, n_embd, n_ff_shared);
        ds3_ref_silu(sg, sg, n_ff_shared);
        ds3_ref_vec_matmul(input, w_shared_up, su, n_embd, n_ff_shared);
        ds3_ref_mul(sg, su, sg, n_ff_shared);
        ds3_ref_vec_matmul(sg, w_shared_down, so, n_ff_shared, n_embd);

        for (int j = 0; j < n_embd; j++) {
            output[j] += so[j];
        }

        free(sg);
        free(su);
        free(so);
    }
}

/* ============================================================================
 * 6. Single-Layer Forward
 * ============================================================================ */

void ds3_ref_layer_forward(
    const ds3_layer_weights_t *layer,
    const float *input,
    ds3_kv_cache_t *kv_cache,
    int seq_pos, int n_tokens,
    float *output,
    ds3_layer_state_t *state)
{
    const int n_embd = DS3_N_EMBD;

    /* Helper: dequantize if tensor exists, else NULL.
     * Reference path allocates internally — correctness over speed. */
    #define DEQ(t_) ((t_) ? ds3_ref_dequantize_tensor(t_) : NULL)
    #define FREE_DEQ(p_) do { if (p_) free((void *)(p_)); } while (0)

    /* --- Dequantize all weights used this layer --- */
    const float *attn_norm   = DEQ(layer->attn_norm);
    const float *attn_q      = DEQ(layer->attn_q);
    const float *attn_q_norm = DS3_HAS_QK_NORM ? DEQ(layer->attn_q_norm) : NULL;
    const float *attn_k      = DEQ(layer->attn_k);
    const float *attn_k_norm = DS3_HAS_QK_NORM ? DEQ(layer->attn_k_norm) : NULL;
    const float *attn_v      = DEQ(layer->attn_v);
    const float *attn_o      = DEQ(layer->attn_output);
    const float *ffn_norm    = DEQ(layer->ffn_norm);
    const float *gate_inp    = DEQ(layer->ffn_gate_inp);
    const float *gate_exps   = DEQ(layer->ffn_gate_exps);
    const float *up_exps     = DEQ(layer->ffn_up_exps);
    const float *down_exps   = DEQ(layer->ffn_down_exps);
    const float *sh_gate     = DEQ(layer->shared_expert_gate);
    const float *sh_up       = DEQ(layer->shared_expert_up);
    const float *sh_down     = DEQ(layer->shared_expert_down);

    /* --- Attention branch --- */
    ds3_ref_rms_norm(input, attn_norm, DS3_NORM_EPS, n_embd, state->xb);

    ds3_ref_gqa_attention(
        state->xb,
        attn_q, attn_q_norm,
        attn_k, attn_k_norm,
        attn_v, attn_o,
        kv_cache, seq_pos, n_tokens,
        state->attn_out,
        state->q, state->k, state->v,
        state->attn_score);

    ds3_ref_add(input, state->attn_out, state->attn_out, n_embd);

    /* --- FFN branch --- */
    ds3_ref_rms_norm(state->attn_out, ffn_norm, DS3_NORM_EPS, n_embd, state->xb);

    ds3_ref_moe_ffn(
        state->xb,
        gate_inp, gate_exps, up_exps, down_exps,
        sh_gate, sh_up, sh_down,
        state->ffn_out,
        state->gate_logits,
        state->expert_weights,
        state->expert_indices,
        n_embd, DS3_N_EXPERT, DS3_N_EXPERT_USED, DS3_N_FF_EXP, DS3_N_FF_SHARED,
        DS3_NORM_TOPK_PROB);

    ds3_ref_add(state->attn_out, state->ffn_out, output, n_embd);

    /* --- Free temporary dequantized buffers --- */
    FREE_DEQ(attn_norm);
    FREE_DEQ(attn_q);
    FREE_DEQ(attn_q_norm);
    FREE_DEQ(attn_k);
    FREE_DEQ(attn_k_norm);
    FREE_DEQ(attn_v);
    FREE_DEQ(attn_o);
    FREE_DEQ(ffn_norm);
    FREE_DEQ(gate_inp);
    FREE_DEQ(gate_exps);
    FREE_DEQ(up_exps);
    FREE_DEQ(down_exps);
    FREE_DEQ(sh_gate);
    FREE_DEQ(sh_up);
    FREE_DEQ(sh_down);

    #undef DEQ
    #undef FREE_DEQ
}
