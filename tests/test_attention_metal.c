/*
 * test_attention_metal.c — Validate Metal decode-only GQA attention kernel.
 *
 * Tests the core attention computation (scores + softmax + weighted sum) in
 * isolation. Projections, RMSNorm and RoPE are tested separately and will be
 * wired together in the end-to-end test (test_e2e_metal).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "src/ds3_metal.h"
#include "src/ds3_reference.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL: %s at %s:%d\n", #x, __FILE__, __LINE__); return 1; } } while(0)

static float max_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void ref_attention_decode(
    const float *q, const float *k, const float *v,
    float *k_cache, float *v_cache,
    uint32_t seq_pos, uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float *output)
{
    const uint32_t seq_len      = seq_pos + 1;
    const uint32_t kv_dim       = n_kv_heads * head_dim;
    const uint32_t heads_per_kv = n_q_heads / n_kv_heads;
    const float scale           = 1.0f / sqrtf((float)head_dim);

    /* Write new K/V into cache at seq_pos (CPU reference uses FP32 cache). */
    for (uint32_t kvh = 0; kvh < n_kv_heads; kvh++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            k_cache[seq_pos * kv_dim + kvh * head_dim + d] = k[kvh * head_dim + d];
            v_cache[seq_pos * kv_dim + kvh * head_dim + d] = v[kvh * head_dim + d];
        }
    }

    for (uint32_t qh = 0; qh < n_q_heads; qh++) {
        const uint32_t kvh     = qh / heads_per_kv;
        const float   *q_head  = q + qh * head_dim;
        float         *out_head = output + qh * head_dim;

        /* Compute scores. */
        float *scores = (float *)malloc(seq_len * sizeof(float));
        if (!scores) {
            fprintf(stderr, "ref_attention_decode: malloc failed\n");
            return;
        }
        for (uint32_t pos = 0; pos < seq_len; pos++) {
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                score += q_head[d] * k_cache[pos * kv_dim + kvh * head_dim + d];
            }
            scores[pos] = score * scale;
        }

        /* Softmax over positions. */
        ds3_ref_softmax(scores, (int)seq_len);

        /* Weighted sum of V. */
        for (uint32_t d = 0; d < head_dim; d++) out_head[d] = 0.0f;
        for (uint32_t pos = 0; pos < seq_len; pos++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                out_head[d] += scores[pos] * v_cache[pos * kv_dim + kvh * head_dim + d];
            }
        }
        free(scores);
    }
}

static int test_attention_decode_small(void) {
    fprintf(stderr, "test_attention_decode_small ... ");

    const uint32_t n_q_heads  = 8;   /* small for fast test */
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim   = 64;  /* <= MAX_HEAD_DIM */
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 16;
    const uint32_t seq_pos    = 7;   /* 8 tokens in cache */

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    /* Pre-fill cache with random data for positions 0..seq_pos-1. */
    srand(60);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    ref_attention_decode(q, k, v, k_cache_ref, v_cache_ref,
                         seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref);

    /* GPU buffers: KV cache is FP16, Q/K/V/output are FP32. */
    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                      seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_attention_decode_qwen3_dims(void) {
    fprintf(stderr, "test_attention_decode_qwen3_dims ... ");

    /* Qwen3-30B attention head configuration. */
    const uint32_t n_q_heads  = 32;
    const uint32_t n_kv_heads = 4;
    const uint32_t head_dim   = 128;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 32;
    const uint32_t seq_pos    = 15;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(61);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    ref_attention_decode(q, k, v, k_cache_ref, v_cache_ref,
                         seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                      seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* SIMD-group parallel variant, using the same Qwen3-30B dimensions. */
static int test_attention_decode_simd(void) {
    fprintf(stderr, "test_attention_decode_simd ... ");

    const uint32_t n_q_heads  = 32;
    const uint32_t n_kv_heads = 4;
    const uint32_t head_dim   = 128;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 64;
    const uint32_t seq_pos    = 31;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(63);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    ref_attention_decode(q, k, v, k_cache_ref, v_cache_ref,
                         seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode_simd(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                           seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Helper: run CPU reference with RoPE applied to Q/K before attention. */
static int ref_attention_decode_rope(
    const float *q, const float *k, const float *v,
    float *k_cache, float *v_cache,
    uint32_t seq_pos, uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float *output)
{
    float *q_rot = malloc((size_t)n_q_heads * head_dim * sizeof(float));
    float *k_rot = malloc((size_t)n_kv_heads * head_dim * sizeof(float));
    if (!q_rot || !k_rot) {
        fprintf(stderr, "ref_attention_decode_rope: malloc failed\n");
        free(q_rot);
        free(k_rot);
        return -1;
    }
    memcpy(q_rot, q, (size_t)n_q_heads * head_dim * sizeof(float));
    memcpy(k_rot, k, (size_t)n_kv_heads * head_dim * sizeof(float));

    ds3_ref_rope(q_rot, k_rot, (int)seq_pos,
                 (int)n_q_heads, (int)n_kv_heads, (int)head_dim);

    ref_attention_decode(q_rot, k_rot, v, k_cache, v_cache,
                         seq_pos, n_q_heads, n_kv_heads, head_dim, output);

    free(q_rot);
    free(k_rot);
    return 0;
}

static int test_attention_decode_rope_small(void) {
    fprintf(stderr, "test_attention_decode_rope_small ... ");

    const uint32_t n_q_heads  = 8;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim   = 64;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 16;
    const uint32_t seq_pos    = 7;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(64);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    CHECK(ref_attention_decode_rope(q, k, v, k_cache_ref, v_cache_ref,
                                    seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref) == 0);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode_rope(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                           seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim,
                                           DS3_ROPE_THETA) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_attention_decode_rope_qwen3_dims(void) {
    fprintf(stderr, "test_attention_decode_rope_qwen3_dims ... ");

    const uint32_t n_q_heads  = 32;
    const uint32_t n_kv_heads = 4;
    const uint32_t head_dim   = 128;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 32;
    const uint32_t seq_pos    = 15;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(65);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    CHECK(ref_attention_decode_rope(q, k, v, k_cache_ref, v_cache_ref,
                                    seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref) == 0);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode_rope(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                           seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim,
                                           DS3_ROPE_THETA) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_attention_decode_rope_qwen3_dims_simd(void) {
    fprintf(stderr, "test_attention_decode_rope_qwen3_dims_simd ... ");

    const uint32_t n_q_heads  = 32;
    const uint32_t n_kv_heads = 4;
    const uint32_t head_dim   = 128;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 32;
    const uint32_t seq_pos    = 15;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(66);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    CHECK(ref_attention_decode_rope(q, k, v, k_cache_ref, v_cache_ref,
                                    seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref) == 0);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode_rope_simd(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                                seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim,
                                                DS3_ROPE_THETA) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Edge case: only one cached position (seq_pos = 0).
 * Softmax over a single value should return v_cache[0] for each head. */
static int test_attention_decode_seq_pos_zero(void) {
    fprintf(stderr, "test_attention_decode_seq_pos_zero ... ");

    const uint32_t n_q_heads  = 8;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim   = 64;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 8;
    const uint32_t seq_pos    = 0;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(62);
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    ref_attention_decode(q, k, v, k_cache_ref, v_cache_ref,
                         seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                      seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Edge case: seq_pos = 0 with integrated RoPE.
 * At position 0, RoPE cos=1 sin=0, so output should equal v_cache[0]. */
static int test_attention_decode_rope_seq_pos_zero(void) {
    fprintf(stderr, "test_attention_decode_rope_seq_pos_zero ... ");

    const uint32_t n_q_heads  = 8;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim   = 64;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 8;
    const uint32_t seq_pos    = 0;

    float *q  = calloc(q_dim, sizeof(float));
    float *k  = calloc(kv_dim, sizeof(float));
    float *v  = calloc(kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(67);
    for (uint32_t i = 0; i < q_dim; i++)  q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < kv_dim; i++) v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    CHECK(ref_attention_decode_rope(q, k, v, k_cache_ref, v_cache_ref,
                                    seq_pos, n_q_heads, n_kv_heads, head_dim, output_ref) == 0);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf  = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(q_dim * sizeof(float));
    CHECK(q_buf && k_buf && v_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf,  0, q,  q_dim  * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,  0, k,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf,  0, v,  kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_decode_rope(q_buf, k_buf, v_buf, kc_buf, vc_buf, out_buf,
                                           seq_pos, max_seq_len, n_q_heads, n_kv_heads, head_dim,
                                           DS3_ROPE_THETA) == 0);

    float *output_mtl = calloc(q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)q_dim);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_kv_cache_write_batch(void) {
    fprintf(stderr, "test_kv_cache_write_batch ... ");

    const uint32_t n_q_heads  = 8;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim   = 64;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 32;
    const uint32_t seq_pos    = 5;
    const uint32_t n_tokens   = 7;

    float *k = calloc(n_tokens * kv_dim, sizeof(float));
    float *v = calloc(n_tokens * kv_dim, sizeof(float));
    CHECK(k && v);

    srand(80);
    for (uint32_t i = 0; i < n_tokens * kv_dim; i++) {
        k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    ds3_metal_buffer_t *k_buf  = ds3_metal_buffer_alloc(n_tokens * kv_dim * sizeof(float));
    ds3_metal_buffer_t *v_buf  = ds3_metal_buffer_alloc(n_tokens * kv_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    CHECK(k_buf && v_buf && kc_buf && vc_buf);

    CHECK(ds3_metal_buffer_write(k_buf, 0, k, n_tokens * kv_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(v_buf, 0, v, n_tokens * kv_dim * sizeof(float)) == 0);

    CHECK(ds3_metal_kv_cache_write_batch(k_buf, v_buf, kc_buf, vc_buf,
                                          seq_pos, n_tokens, max_seq_len,
                                          n_q_heads, n_kv_heads, head_dim) == 0);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    CHECK(ds3_metal_buffer_read(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_read(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    float diff = 0.0f;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < kv_dim; i++) {
            _Float16 kh, vh;
            memcpy(&kh, &k_cache_half[(seq_pos + t) * kv_dim + i], sizeof(uint16_t));
            memcpy(&vh, &v_cache_half[(seq_pos + t) * kv_dim + i], sizeof(uint16_t));
            float dk = fabsf((float)kh - k[t * kv_dim + i]);
            float dv = fabsf((float)vh - v[t * kv_dim + i]);
            if (dk > diff) diff = dk;
            if (dv > diff) diff = dv;
        }
    }
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(v_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    free(k); free(v);
    free(k_cache_half); free(v_cache_half);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static void ref_attention_chunk(
    const float *q,
    const float *k_cache, const float *v_cache,
    uint32_t seq_pos, uint32_t n_tokens,
    uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float *output)
{
    const uint32_t kv_dim       = n_kv_heads * head_dim;
    const uint32_t q_dim        = n_q_heads * head_dim;
    const uint32_t heads_per_kv = n_q_heads / n_kv_heads;
    const float scale           = 1.0f / sqrtf((float)head_dim);

    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t total_len = seq_pos + t + 1;
        for (uint32_t qh = 0; qh < n_q_heads; qh++) {
            const uint32_t kvh      = qh / heads_per_kv;
            const float   *q_head   = q + t * q_dim + qh * head_dim;
            float         *out_head = output + t * q_dim + qh * head_dim;

            float max_score = -1e30f;
            for (uint32_t pos = 0; pos < total_len; pos++) {
                const float *k_head = k_cache + pos * kv_dim + kvh * head_dim;
                float score = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) score += q_head[d] * k_head[d];
                score *= scale;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            float out_acc[128];
            for (uint32_t d = 0; d < head_dim; d++) out_acc[d] = 0.0f;

            for (uint32_t pos = 0; pos < total_len; pos++) {
                const float *k_head = k_cache + pos * kv_dim + kvh * head_dim;
                const float *v_head = v_cache + pos * kv_dim + kvh * head_dim;
                float score = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) score += q_head[d] * k_head[d];
                score *= scale;
                float w = expf(score - max_score);
                sum_exp += w;
                for (uint32_t d = 0; d < head_dim; d++) out_acc[d] += w * v_head[d];
            }

            float inv = 1.0f / sum_exp;
            for (uint32_t d = 0; d < head_dim; d++) out_head[d] = out_acc[d] * inv;
        }
    }
}

static int test_attention_chunk_small(void) {
    fprintf(stderr, "test_attention_chunk_small ... ");

    const uint32_t n_q_heads  = 8;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim   = 64;
    const uint32_t q_dim      = n_q_heads * head_dim;
    const uint32_t kv_dim     = n_kv_heads * head_dim;
    const uint32_t max_seq_len = 32;
    const uint32_t seq_pos    = 6;
    const uint32_t n_tokens   = 5;

    float *q  = calloc(n_tokens * q_dim, sizeof(float));
    float *k  = calloc(n_tokens * kv_dim, sizeof(float));
    float *v  = calloc(n_tokens * kv_dim, sizeof(float));
    float *k_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *v_cache_ref = calloc(max_seq_len * kv_dim, sizeof(float));
    float *output_ref  = calloc(n_tokens * q_dim, sizeof(float));
    CHECK(q && k && v && k_cache_ref && v_cache_ref && output_ref);

    srand(81);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        k_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache_ref[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < n_tokens * q_dim; i++) q[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_tokens * kv_dim; i++) {
        k[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    /* Merge the chunk K/V into the reference cache so the reference reads the
     * same FP16-rounded values the kernel will read from the cache. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        memcpy(k_cache_ref + (seq_pos + t) * kv_dim, k + t * kv_dim, kv_dim * sizeof(float));
        memcpy(v_cache_ref + (seq_pos + t) * kv_dim, v + t * kv_dim, kv_dim * sizeof(float));
    }

    ref_attention_chunk(q, k_cache_ref, v_cache_ref,
                        seq_pos, n_tokens, n_q_heads, n_kv_heads, head_dim, output_ref);

    uint16_t *k_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    uint16_t *v_cache_half = calloc(max_seq_len * kv_dim, sizeof(uint16_t));
    CHECK(k_cache_half && v_cache_half);
    for (uint32_t i = 0; i < max_seq_len * kv_dim; i++) {
        _Float16 kh = (_Float16)k_cache_ref[i];
        _Float16 vh = (_Float16)v_cache_ref[i];
        memcpy(&k_cache_half[i], &kh, sizeof(uint16_t));
        memcpy(&v_cache_half[i], &vh, sizeof(uint16_t));
    }

    ds3_metal_buffer_t *q_buf   = ds3_metal_buffer_alloc(n_tokens * q_dim * sizeof(float));
    ds3_metal_buffer_t *kc_buf  = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *vc_buf  = ds3_metal_buffer_alloc(max_seq_len * kv_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *out_buf = ds3_metal_buffer_alloc(n_tokens * q_dim * sizeof(float));
    CHECK(q_buf && kc_buf && vc_buf && out_buf);

    CHECK(ds3_metal_buffer_write(q_buf, 0, q, n_tokens * q_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(kc_buf, 0, k_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(vc_buf, 0, v_cache_half, max_seq_len * kv_dim * sizeof(uint16_t)) == 0);

    CHECK(ds3_metal_attention_chunk(q_buf, kc_buf, vc_buf, out_buf,
                                     seq_pos, n_tokens, max_seq_len,
                                     n_q_heads, n_kv_heads, head_dim) == 0);

    float *output_mtl = calloc(n_tokens * q_dim, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, n_tokens * q_dim * sizeof(float)) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)(n_tokens * q_dim));
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(kc_buf);
    ds3_metal_buffer_free(vc_buf);
    ds3_metal_buffer_free(out_buf);
    free(q); free(k); free(v);
    free(k_cache_ref); free(v_cache_ref); free(output_ref);
    free(k_cache_half); free(v_cache_half); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    if (ds3_metal_init() != 0) {
        fprintf(stderr, "Metal init failed, skipping tests.\n");
        return 77;
    }

    int failed = 0;
    failed |= test_attention_decode_small();
    failed |= test_attention_decode_qwen3_dims();
    failed |= test_attention_decode_simd();
    failed |= test_attention_decode_seq_pos_zero();
    failed |= test_attention_decode_rope_small();
    failed |= test_attention_decode_rope_qwen3_dims();
    failed |= test_attention_decode_rope_qwen3_dims_simd();
    failed |= test_attention_decode_rope_seq_pos_zero();
    failed |= test_kv_cache_write_batch();
    failed |= test_attention_chunk_small();

    ds3_metal_shutdown();

    if (failed) {
        fprintf(stderr, "\n%d test(s) FAILED\n", failed);
        return 1;
    }
    fprintf(stderr, "\nAll attention tests PASSED\n");
    return 0;
}
