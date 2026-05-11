/*
 * test_metal.c — Validate Metal kernels against CPU FP32 reference.
 *
 * Tests:
 *   1. rms_norm_f32  vs ds3_ref_rms_norm
 *   2. rope_f32      vs ds3_ref_rope
 *
 * Pass criterion: max_diff < 1e-4 for all elements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

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

static int test_rms_norm(void) {
    fprintf(stderr, "test_rms_norm ... ");

    const int n      = 2048;   // Qwen3 n_embd (30B)
    const int n_rows = 4;      // batch test
    const float eps  = 1e-6f;

    float *x_host      = calloc(n_rows * n, sizeof(float));
    float *weight_host = calloc(n, sizeof(float));
    float *ref_out     = calloc(n_rows * n, sizeof(float));
    float *mtl_out     = calloc(n_rows * n, sizeof(float));
    CHECK(x_host && weight_host && ref_out && mtl_out);

    srand(42);
    for (int i = 0; i < n_rows * n; i++) x_host[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < n; i++) weight_host[i] = 1.0f + (float)(rand() % 100) / 1000.0f;

    /* CPU reference (row by row). */
    for (int r = 0; r < n_rows; r++) {
        ds3_ref_rms_norm(x_host + r * n, weight_host, eps, n, ref_out + r * n);
    }

    /* Metal. */
    ds3_metal_buffer_t *x_buf      = ds3_metal_buffer_alloc(n_rows * n * sizeof(float));
    ds3_metal_buffer_t *w_buf      = ds3_metal_buffer_alloc(n * sizeof(float));
    ds3_metal_buffer_t *out_buf    = ds3_metal_buffer_alloc(n_rows * n * sizeof(float));
    CHECK(x_buf && w_buf && out_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x_host, n_rows * n * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(w_buf, 0, weight_host, n * sizeof(float)) == 0);

    CHECK(ds3_metal_rms_norm(x_buf, w_buf, out_buf, n, n_rows, eps) == 0);

    CHECK(ds3_metal_buffer_read(out_buf, 0, mtl_out, n_rows * n * sizeof(float)) == 0);

    float diff = max_diff(ref_out, mtl_out, n_rows * n);
    if (diff >= 1e-4f) {
        fprintf(stderr, "FAIL (max_diff=%.6e)\n", diff);
        /* Print first few mismatches for debugging. */
        int printed = 0;
        for (int i = 0; i < n_rows * n && printed < 5; i++) {
            float d = fabsf(ref_out[i] - mtl_out[i]);
            if (d >= 1e-4f) {
                fprintf(stderr, "  [%d] ref=%.6f mtl=%.6f diff=%.6e\n",
                        i, ref_out[i], mtl_out[i], d);
                printed++;
            }
        }
        return 1;
    }

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(w_buf);
    ds3_metal_buffer_free(out_buf);
    free(x_host); free(weight_host); free(ref_out); free(mtl_out);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_rope(void) {
    fprintf(stderr, "test_rope ... ");

    const int n_heads    = 32;   // Qwen3-30B
    const int n_kv_heads = 4;
    const int head_dim   = 128;
    const int n_rows     = 4;    // batch test (different seq positions)
    const float theta    = 1e6f;

    int total_q = n_rows * n_heads * head_dim;
    int total_k = n_rows * n_kv_heads * head_dim;

    float *q_host  = calloc(total_q, sizeof(float));
    float *k_host  = calloc(total_k, sizeof(float));
    float *q_ref   = calloc(total_q, sizeof(float));
    float *k_ref   = calloc(total_k, sizeof(float));
    float *q_mtl   = calloc(total_q, sizeof(float));
    float *k_mtl   = calloc(total_k, sizeof(float));
    int32_t *pos   = calloc(n_rows, sizeof(int32_t));
    CHECK(q_host && k_host && q_ref && k_ref && q_mtl && k_mtl && pos);

    srand(43);
    for (int i = 0; i < total_q; i++) q_host[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < total_k; i++) k_host[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int r = 0; r < n_rows; r++) pos[r] = r * 7;  // positions: 0, 7, 14, 21

    /* CPU reference (row by row, in-place). */
    memcpy(q_ref, q_host, total_q * sizeof(float));
    memcpy(k_ref, k_host, total_k * sizeof(float));
    for (int r = 0; r < n_rows; r++) {
        ds3_ref_rope(q_ref + r * n_heads * head_dim,
                     k_ref + r * n_kv_heads * head_dim,
                     pos[r], n_heads, n_kv_heads, head_dim);
    }

    /* Metal: out-of-place, so copy input to output buffer first. */
    ds3_metal_buffer_t *q_buf = ds3_metal_buffer_alloc(total_q * sizeof(float));
    ds3_metal_buffer_t *k_buf = ds3_metal_buffer_alloc(total_k * sizeof(float));
    ds3_metal_buffer_t *q_out = ds3_metal_buffer_alloc(total_q * sizeof(float));
    ds3_metal_buffer_t *k_out = ds3_metal_buffer_alloc(total_k * sizeof(float));
    ds3_metal_buffer_t *pos_buf = ds3_metal_buffer_alloc(n_rows * sizeof(int32_t));
    CHECK(q_buf && k_buf && q_out && k_out && pos_buf);

    CHECK(ds3_metal_buffer_write(q_buf,   0, q_host, total_q * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_buf,   0, k_host, total_k * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(q_out,   0, q_host, total_q * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(k_out,   0, k_host, total_k * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(pos_buf, 0, pos,    n_rows  * sizeof(int32_t)) == 0);

    CHECK(ds3_metal_rope(q_buf, q_out, pos_buf, n_heads, head_dim, n_rows, theta) == 0);
    CHECK(ds3_metal_rope(k_buf, k_out, pos_buf, n_kv_heads, head_dim, n_rows, theta) == 0);

    CHECK(ds3_metal_buffer_read(q_out, 0, q_mtl, total_q * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_read(k_out, 0, k_mtl, total_k * sizeof(float)) == 0);

    float diff_q = max_diff(q_ref, q_mtl, total_q);
    float diff_k = max_diff(k_ref, k_mtl, total_k);
    if (diff_q >= 1e-4f || diff_k >= 1e-4f) {
        fprintf(stderr, "FAIL (diff_q=%.6e diff_k=%.6e)\n", diff_q, diff_k);
        int printed = 0;
        for (int i = 0; i < total_q && printed < 5; i++) {
            float d = fabsf(q_ref[i] - q_mtl[i]);
            if (d >= 1e-4f) {
                fprintf(stderr, "  q[%d] ref=%.6f mtl=%.6f diff=%.6e\n",
                        i, q_ref[i], q_mtl[i], d);
                printed++;
            }
        }
        return 1;
    }

    ds3_metal_buffer_free(q_buf);
    ds3_metal_buffer_free(k_buf);
    ds3_metal_buffer_free(q_out);
    ds3_metal_buffer_free(k_out);
    ds3_metal_buffer_free(pos_buf);
    free(q_host); free(k_host); free(q_ref); free(k_ref); free(q_mtl); free(k_mtl); free(pos);

    fprintf(stderr, "PASS (max_diff_q=%.2e max_diff_k=%.2e)\n", diff_q, diff_k);
    return 0;
}

static int test_matmul(void) {
    fprintf(stderr, "test_matmul ... ");

    const int M = 64;
    const int N = 128;
    const int K = 256;

    float *A = calloc(M * K, sizeof(float));
    float *B = calloc(K * N, sizeof(float));
    float *C_ref = calloc(M * N, sizeof(float));
    float *C_mtl = calloc(M * N, sizeof(float));
    CHECK(A && B && C_ref && C_mtl);

    srand(44);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    /* CPU reference */
    ds3_ref_matmul(A, B, C_ref, M, N, K);

    /* Metal */
    ds3_metal_buffer_t *a_buf = ds3_metal_buffer_alloc(M * K * sizeof(float));
    ds3_metal_buffer_t *b_buf = ds3_metal_buffer_alloc(K * N * sizeof(float));
    ds3_metal_buffer_t *c_buf = ds3_metal_buffer_alloc(M * N * sizeof(float));
    CHECK(a_buf && b_buf && c_buf);

    CHECK(ds3_metal_buffer_write(a_buf, 0, A, M * K * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(b_buf, 0, B, K * N * sizeof(float)) == 0);

    CHECK(ds3_metal_matmul(a_buf, b_buf, c_buf, M, N, K) == 0);

    CHECK(ds3_metal_buffer_read(c_buf, 0, C_mtl, M * N * sizeof(float)) == 0);

    float diff = max_diff(C_ref, C_mtl, M * N);
    if (diff >= 1e-4f) {
        fprintf(stderr, "FAIL (max_diff=%.6e)\n", diff);
        int printed = 0;
        for (int i = 0; i < M * N && printed < 5; i++) {
            float d = fabsf(C_ref[i] - C_mtl[i]);
            if (d >= 1e-4f) {
                fprintf(stderr, "  [%d] ref=%.6f mtl=%.6f diff=%.6e\n",
                        i, C_ref[i], C_mtl[i], d);
                printed++;
            }
        }
        return 1;
    }

    ds3_metal_buffer_free(a_buf);
    ds3_metal_buffer_free(b_buf);
    ds3_metal_buffer_free(c_buf);
    free(A); free(B); free(C_ref); free(C_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_elementwise(void) {
    fprintf(stderr, "test_elementwise ... ");

    const int n = 2048;
    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *c = calloc(n, sizeof(float));
    float *mtl = calloc(n, sizeof(float));
    CHECK(a && b && c && mtl);

    srand(45);
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    ds3_metal_buffer_t *buf_a = ds3_metal_buffer_alloc(n * sizeof(float));
    ds3_metal_buffer_t *buf_b = ds3_metal_buffer_alloc(n * sizeof(float));
    ds3_metal_buffer_t *buf_c = ds3_metal_buffer_alloc(n * sizeof(float));
    CHECK(buf_a && buf_b && buf_c);

    CHECK(ds3_metal_buffer_write(buf_a, 0, a, n * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(buf_b, 0, b, n * sizeof(float)) == 0);

    /* copy: c = a */
    CHECK(ds3_metal_buffer_copy_f32(buf_a, buf_c, n * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_read(buf_c, 0, mtl, n * sizeof(float)) == 0);
    CHECK(max_diff(a, mtl, n) < 1e-6f);

    /* zero: c = 0 */
    CHECK(ds3_metal_buffer_zero_f32(buf_c, n * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_read(buf_c, 0, mtl, n * sizeof(float)) == 0);
    for (int i = 0; i < n; i++) CHECK(fabsf(mtl[i]) < 1e-6f);

    /* add_inplace: a += b */
    CHECK(ds3_metal_buffer_write(buf_a, 0, a, n * sizeof(float)) == 0);
    CHECK(ds3_metal_vec_add_inplace(buf_a, buf_b, n) == 0);
    CHECK(ds3_metal_buffer_read(buf_a, 0, mtl, n * sizeof(float)) == 0);
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    CHECK(max_diff(c, mtl, n) < 1e-6f);

    /* add3: c = a + b (buf_a currently holds a+b from the inplace test) */
    for (int i = 0; i < n; i++) c[i] = (a[i] + b[i]) + b[i];
    CHECK(ds3_metal_vec_add(buf_a, buf_b, buf_c, n) == 0);
    CHECK(ds3_metal_buffer_read(buf_c, 0, mtl, n * sizeof(float)) == 0);
    CHECK(max_diff(c, mtl, n) < 1e-6f);

    ds3_metal_buffer_free(buf_a);
    ds3_metal_buffer_free(buf_b);
    ds3_metal_buffer_free(buf_c);
    free(a); free(b); free(c); free(mtl);

    fprintf(stderr, "PASS\n");
    return 0;
}

static int test_batch_mode(void) {
    fprintf(stderr, "test_batch_mode ... ");

    const int n = 2048;
    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *ref = calloc(n, sizeof(float));
    float *mtl = calloc(n, sizeof(float));
    CHECK(a && b && ref && mtl);

    srand(123);
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        ref[i] = a[i] + b[i] + a[i];  /* (a + b) + a */
    }

    ds3_metal_buffer_t *buf_a = ds3_metal_buffer_alloc(n * sizeof(float));
    ds3_metal_buffer_t *buf_b = ds3_metal_buffer_alloc(n * sizeof(float));
    ds3_metal_buffer_t *buf_out = ds3_metal_buffer_alloc(n * sizeof(float));
    CHECK(buf_a && buf_b && buf_out);
    CHECK(ds3_metal_buffer_write(buf_a, 0, a, n * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(buf_b, 0, b, n * sizeof(float)) == 0);

    CHECK(ds3_metal_begin_batch() == 0);
    CHECK(ds3_metal_vec_add(buf_a, buf_b, buf_out, n) == 0);      /* out = a + b */
    CHECK(ds3_metal_vec_add_inplace(buf_out, buf_a, n) == 0);     /* out = out + a */
    ds3_metal_end_batch();

    CHECK(ds3_metal_buffer_read(buf_out, 0, mtl, n * sizeof(float)) == 0);
    CHECK(max_diff(ref, mtl, n) < 1e-6f);

    ds3_metal_buffer_free(buf_a);
    ds3_metal_buffer_free(buf_b);
    ds3_metal_buffer_free(buf_out);
    free(a); free(b); free(ref); free(mtl);

    fprintf(stderr, "PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    if (ds3_metal_init() != 0) {
        fprintf(stderr, "Metal init failed, skipping tests.\n");
        return 77;  /* skip exit code */
    }

    int failed = 0;
    failed |= test_rms_norm();
    failed |= test_rope();
    failed |= test_matmul();
    failed |= test_elementwise();
    failed |= test_batch_mode();

    ds3_metal_shutdown();

    if (failed) {
        fprintf(stderr, "\n%d test(s) FAILED\n", failed);
        return 1;
    }
    fprintf(stderr, "\nAll Metal kernel tests PASSED\n");
    return 0;
}
