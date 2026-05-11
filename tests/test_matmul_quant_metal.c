/*
 * test_matmul_quant_metal.c — Validate Metal vec-matmul kernels (FP32, Q8_0, Q4_K, Q6_K).
 *
 * Compares GPU outputs against CPU reference:
 *   y_ref = x @ W_f32^T
 *   y_mtl = GPU kernel on W_quant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "src/ds3_metal.h"
#include "src/ds3_reference.h"

#include "tests/test_quant_utils.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL: %s at %s:%d\n", #x, __FILE__, __LINE__); return 1; } } while(0)

static float max_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ============================================================================
 * FP32 baseline
 * ============================================================================ */

static int test_vec_matmul_f32(void) {
    fprintf(stderr, "test_vec_matmul_f32 ... ");

    const int in_dim  = 2048;
    const int out_dim = 4096;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(42);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    ds3_ref_vec_matmul(x, W, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * in_dim * sizeof(float));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W, out_dim * in_dim * sizeof(float)) == 0);

    CHECK(ds3_metal_vec_matmul_f32(x_buf, W_buf, y_buf, in_dim, out_dim,
                                    in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Vector × matrix with non-dense row stride (FP32). */
static int test_vec_matmul_f32_strided(void) {
    fprintf(stderr, "test_vec_matmul_f32_strided ... ");

    const int in_dim     = 64;
    const int out_dim    = 8;
    const int row_stride = 128; /* extra padding per row */

    float *x          = calloc(in_dim, sizeof(float));
    float *W_dense    = calloc(out_dim * in_dim, sizeof(float));
    float *W_strided  = calloc(out_dim * row_stride, sizeof(float));
    float *y_ref      = calloc(out_dim, sizeof(float));
    float *y_mtl      = calloc(out_dim, sizeof(float));
    CHECK(x && W_dense && W_strided && y_ref && y_mtl);

    srand(45);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int r = 0; r < out_dim; r++) {
        for (int c = 0; c < in_dim; c++) {
            float v = (float)(rand() % 1000) / 1000.0f - 0.5f;
            W_dense[r * in_dim + c]       = v;
            W_strided[r * row_stride + c] = v;
        }
        /* poison the padding so the kernel must skip it */
        for (int c = in_dim; c < row_stride; c++) {
            W_strided[r * row_stride + c] = 12345.0f;
        }
    }

    ds3_ref_vec_matmul(x, W_dense, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * row_stride * sizeof(float));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_strided, out_dim * row_stride * sizeof(float)) == 0);

    CHECK(ds3_metal_vec_matmul_f32(x_buf, W_buf, y_buf, in_dim, out_dim,
                                    row_stride * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W_dense); free(W_strided); free(y_ref); free(y_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================
 * Q8_0 quantized
 * ============================================================================ */

static int test_vec_matmul_q8_0(void) {
    fprintf(stderr, "test_vec_matmul_q8_0 ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(43);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / 32;
    ds3_test_block_q8_0 *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q8_0));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q8_0(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    /* CPU reference uses the SAME quantized weights, dequantized to FP32. */
    float *W_deq = calloc(out_dim * in_dim, sizeof(float));
    CHECK(W_deq);
    for (int r = 0; r < out_dim; r++) {
        const ds3_test_block_q8_0 *blk = W_q + r * nb_per_row;
        for (int b = 0; b < nb_per_row; b++) {
            float d = test_fp16_to_fp32(blk[b].d);
            for (int i = 0; i < 32; i++) {
                W_deq[r * in_dim + b * 32 + i] = d * (float)blk[b].qs[i];
            }
        }
    }
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q8_0));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q8_0)) == 0);

    CHECK(ds3_metal_vec_matmul_quantized(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         nb_per_row * sizeof(ds3_test_block_q8_0),
                                         DS3_TYPE_Q8_0) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    free(W_deq);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================
 * Q4_K quantized
 * ============================================================================ */

static int test_vec_matmul_q4_k(void) {
    fprintf(stderr, "test_vec_matmul_q4_k ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(44);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / Q4K_BLOCK_SIZE;
    ds3_test_block_q4_K *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q4_K));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q4_K(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    /* CPU reference uses the SAME quantized weights, dequantized to FP32.
     * Reuse the reference dequantizer to avoid duplicating scale decoding. */
    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q4_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = in_dim;
    W_tensor.ne[1] = out_dim;
    W_tensor.data = W_q;
    W_tensor.size = out_dim * nb_per_row * sizeof(ds3_test_block_q4_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q4_K));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);

    CHECK(ds3_metal_vec_matmul_quantized(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         nb_per_row * sizeof(ds3_test_block_q4_K),
                                         DS3_TYPE_Q4_K) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================
 * SIMD-group parallel reduction variants
 * ============================================================================ */

static int test_vec_matmul_f32_simd(void) {
    fprintf(stderr, "test_vec_matmul_f32_simd ... ");

    const int in_dim  = 2048;
    const int out_dim = 4096;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(46);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    ds3_ref_vec_matmul(x, W, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * in_dim * sizeof(float));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W, out_dim * in_dim * sizeof(float)) == 0);

    CHECK(ds3_metal_vec_matmul_f32_simd(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_vec_matmul_q8_0_simd(void) {
    fprintf(stderr, "test_vec_matmul_q8_0_simd ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(48);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / 32;
    ds3_test_block_q8_0 *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q8_0));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q8_0(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    float *W_deq = calloc(out_dim * in_dim, sizeof(float));
    CHECK(W_deq);
    for (int r = 0; r < out_dim; r++) {
        const ds3_test_block_q8_0 *blk = W_q + r * nb_per_row;
        for (int b = 0; b < nb_per_row; b++) {
            float d = test_fp16_to_fp32(blk[b].d);
            for (int i = 0; i < 32; i++) {
                W_deq[r * in_dim + b * 32 + i] = d * (float)blk[b].qs[i];
            }
        }
    }
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q8_0));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q8_0)) == 0);

    CHECK(ds3_metal_vec_matmul_q8_0_simd(x_buf, W_buf, y_buf, in_dim, out_dim,
                                          nb_per_row * sizeof(ds3_test_block_q8_0)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    free(W_deq);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_vec_matmul_q4k_simd(void) {
    fprintf(stderr, "test_vec_matmul_q4k_simd ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(49);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / Q4K_BLOCK_SIZE;
    ds3_test_block_q4_K *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q4_K));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q4_K(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q4_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = in_dim;
    W_tensor.ne[1] = out_dim;
    W_tensor.data = W_q;
    W_tensor.size = out_dim * nb_per_row * sizeof(ds3_test_block_q4_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q4_K));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);

    CHECK(ds3_metal_vec_matmul_q4k_simd(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================
 * Q6_K quantized
 * ============================================================================ */

static int test_vec_matmul_q6_k(void) {
    fprintf(stderr, "test_vec_matmul_q6_k ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(51);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / Q6K_BLOCK_SIZE;
    ds3_test_block_q6_K *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q6_K));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q6_K(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q6_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = in_dim;
    W_tensor.ne[1] = out_dim;
    W_tensor.data = W_q;
    W_tensor.size = out_dim * nb_per_row * sizeof(ds3_test_block_q6_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q6_K));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q6_K)) == 0);

    CHECK(ds3_metal_vec_matmul_quantized(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         nb_per_row * sizeof(ds3_test_block_q6_K),
                                         DS3_TYPE_Q6_K) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_vec_matmul_q6k_simd(void) {
    fprintf(stderr, "test_vec_matmul_q6k_simd ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(52);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / Q6K_BLOCK_SIZE;
    ds3_test_block_q6_K *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q6_K));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q6_K(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q6_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = in_dim;
    W_tensor.ne[1] = out_dim;
    W_tensor.data = W_q;
    W_tensor.size = out_dim * nb_per_row * sizeof(ds3_test_block_q6_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q6_K));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x, in_dim * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q6_K)) == 0);

    CHECK(ds3_metal_vec_matmul_q6k_simd(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         nb_per_row * sizeof(ds3_test_block_q6_K)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================
 * Half-precision input variant (Q4_K)
 * ============================================================================ */

static int test_vec_matmul_q4k_half(void) {
    fprintf(stderr, "test_vec_matmul_q4k_half ... ");

    const int in_dim  = 2048;
    const int out_dim = 1024;

    float *x  = calloc(in_dim, sizeof(float));
    float *W  = calloc(out_dim * in_dim, sizeof(float));
    float *y_ref = calloc(out_dim, sizeof(float));
    float *y_mtl = calloc(out_dim, sizeof(float));
    CHECK(x && W && y_ref && y_mtl);

    srand(50);
    for (int i = 0; i < in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = in_dim / Q4K_BLOCK_SIZE;
    ds3_test_block_q4_K *W_q = calloc(out_dim * nb_per_row, sizeof(ds3_test_block_q4_K));
    CHECK(W_q);
    for (int r = 0; r < out_dim; r++) {
        quantize_row_q4_K(W + r * in_dim, W_q + r * nb_per_row, in_dim);
    }

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q4_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = in_dim;
    W_tensor.ne[1] = out_dim;
    W_tensor.data = W_q;
    W_tensor.size = out_dim * nb_per_row * sizeof(ds3_test_block_q4_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    /* The half-input kernel reads x as FP16. Build a CPU reference that uses the
     * same FP16-rounded values so we only compare kernel arithmetic, not the
     * FP16 quantization error of the input itself. */
    uint16_t *x_half = calloc(in_dim, sizeof(uint16_t));
    CHECK(x_half);
    for (int i = 0; i < in_dim; i++) {
        _Float16 h = (_Float16)x[i];
        memcpy(&x_half[i], &h, sizeof(uint16_t));
        x[i] = (float)h; /* overwrite reference input with rounded value */
    }
    ds3_ref_vec_matmul(x, W_deq, y_ref, in_dim, out_dim);

    ds3_metal_buffer_t *x_buf = ds3_metal_buffer_alloc(in_dim * sizeof(uint16_t));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(out_dim * nb_per_row * sizeof(ds3_test_block_q4_K));
    ds3_metal_buffer_t *y_buf = ds3_metal_buffer_alloc(out_dim * sizeof(float));
    CHECK(x_buf && W_buf && y_buf);

    CHECK(ds3_metal_buffer_write(x_buf, 0, x_half, in_dim * sizeof(uint16_t)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, out_dim * nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);

    CHECK(ds3_metal_vec_matmul_q4k_half(x_buf, W_buf, y_buf, in_dim, out_dim,
                                         nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);
    CHECK(ds3_metal_buffer_read(y_buf, 0, y_mtl, out_dim * sizeof(float)) == 0);

    float diff = max_diff(y_ref, y_mtl, out_dim);
    CHECK(diff < 1e-4f);

    ds3_metal_buffer_free(x_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(y_buf);
    free(x); free(W); free(y_ref); free(y_mtl); free(W_q); free(W_deq); free(x_half);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================
 * Batched quantized matrix × matrix (prefill-style)
 * ============================================================================ */

static int test_batch_matmul_q8_0(void) {
    fprintf(stderr, "test_batch_matmul_q8_0 ... ");

    const int M = 32;
    const int N = 128;
    const int K = 256;

    float *A = calloc(M * K, sizeof(float));
    float *W = calloc(N * K, sizeof(float));
    float *C_ref = calloc(M * N, sizeof(float));
    float *C_mtl = calloc(M * N, sizeof(float));
    CHECK(A && W && C_ref && C_mtl);

    srand(71);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < N * K; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = K / 32;
    ds3_test_block_q8_0 *W_q = calloc(N * nb_per_row, sizeof(ds3_test_block_q8_0));
    CHECK(W_q);
    for (int r = 0; r < N; r++) quantize_row_q8_0(W + r * K, W_q + r * nb_per_row, K);

    float *W_deq = calloc(N * K, sizeof(float));
    CHECK(W_deq);
    for (int r = 0; r < N; r++) {
        const ds3_test_block_q8_0 *blk = W_q + r * nb_per_row;
        for (int b = 0; b < nb_per_row; b++) {
            float d = test_fp16_to_fp32(blk[b].d);
            for (int i = 0; i < 32; i++) {
                W_deq[r * K + b * 32 + i] = d * (float)blk[b].qs[i];
            }
        }
    }
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * W_deq[n * K + k];
            C_ref[m * N + n] = sum;
        }
    }

    ds3_metal_buffer_t *A_buf = ds3_metal_buffer_alloc(M * K * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(N * nb_per_row * sizeof(ds3_test_block_q8_0));
    ds3_metal_buffer_t *C_buf = ds3_metal_buffer_alloc(M * N * sizeof(float));
    CHECK(A_buf && W_buf && C_buf);

    CHECK(ds3_metal_buffer_write(A_buf, 0, A, M * K * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, N * nb_per_row * sizeof(ds3_test_block_q8_0)) == 0);

    CHECK(ds3_metal_matmul_quantized_batch(A_buf, W_buf, C_buf,
                                            M, N, K,
                                            nb_per_row * sizeof(ds3_test_block_q8_0),
                                            0, DS3_TYPE_Q8_0) == 0);
    CHECK(ds3_metal_buffer_read(C_buf, 0, C_mtl, M * N * sizeof(float)) == 0);

    float diff = max_diff(C_ref, C_mtl, M * N);
    CHECK(diff < 2e-4f);

    ds3_metal_buffer_free(A_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(C_buf);
    free(A); free(W); free(C_ref); free(C_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_batch_matmul_q4_k(void) {
    fprintf(stderr, "test_batch_matmul_q4_k ... ");

    const int M = 16;
    const int N = 64;
    const int K = 256;

    float *A = calloc(M * K, sizeof(float));
    float *W = calloc(N * K, sizeof(float));
    float *C_ref = calloc(M * N, sizeof(float));
    float *C_mtl = calloc(M * N, sizeof(float));
    CHECK(A && W && C_ref && C_mtl);

    srand(72);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < N * K; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = K / Q4K_BLOCK_SIZE;
    ds3_test_block_q4_K *W_q = calloc(N * nb_per_row, sizeof(ds3_test_block_q4_K));
    CHECK(W_q);
    for (int r = 0; r < N; r++) quantize_row_q4_K(W + r * K, W_q + r * nb_per_row, K);

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q4_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = K;
    W_tensor.ne[1] = N;
    W_tensor.data = W_q;
    W_tensor.size = N * nb_per_row * sizeof(ds3_test_block_q4_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * W_deq[n * K + k];
            C_ref[m * N + n] = sum;
        }
    }

    ds3_metal_buffer_t *A_buf = ds3_metal_buffer_alloc(M * K * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(N * nb_per_row * sizeof(ds3_test_block_q4_K));
    ds3_metal_buffer_t *C_buf = ds3_metal_buffer_alloc(M * N * sizeof(float));
    CHECK(A_buf && W_buf && C_buf);

    CHECK(ds3_metal_buffer_write(A_buf, 0, A, M * K * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, N * nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);

    CHECK(ds3_metal_matmul_quantized_batch(A_buf, W_buf, C_buf,
                                            M, N, K,
                                            nb_per_row * sizeof(ds3_test_block_q4_K),
                                            0, DS3_TYPE_Q4_K) == 0);
    CHECK(ds3_metal_buffer_read(C_buf, 0, C_mtl, M * N * sizeof(float)) == 0);

    float diff = max_diff(C_ref, C_mtl, M * N);
    CHECK(diff < 2e-4f);

    ds3_metal_buffer_free(A_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(C_buf);
    free(A); free(W); free(C_ref); free(C_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Compare batched and vector Q4_K matmuls for the M == 1 decode case.
 * This catches any divergence between the chunk-prefill and decode paths. */
static int test_batch_vs_vec_q4_k(void) {
    fprintf(stderr, "test_batch_vs_vec_q4_k ... ");

    const int M = 1;
    const int N = 4096;
    const int K = 2048;

    float *A = calloc(M * K, sizeof(float));
    float *W = calloc(N * K, sizeof(float));
    float *C_ref = calloc(M * N, sizeof(float));
    float *C_vec = calloc(M * N, sizeof(float));
    float *C_batch = calloc(M * N, sizeof(float));
    CHECK(A && W && C_ref && C_vec && C_batch);

    srand(99);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < N * K; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = K / Q4K_BLOCK_SIZE;
    ds3_test_block_q4_K *W_q = calloc(N * nb_per_row, sizeof(ds3_test_block_q4_K));
    CHECK(W_q);
    for (int r = 0; r < N; r++) quantize_row_q4_K(W + r * K, W_q + r * nb_per_row, K);

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q4_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = K;
    W_tensor.ne[1] = N;
    W_tensor.data = W_q;
    W_tensor.size = N * nb_per_row * sizeof(ds3_test_block_q4_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * W_deq[n * K + k];
            C_ref[m * N + n] = sum;
        }
    }

    ds3_metal_buffer_t *A_buf   = ds3_metal_buffer_alloc(M * K * sizeof(float));
    ds3_metal_buffer_t *W_buf   = ds3_metal_buffer_alloc(N * nb_per_row * sizeof(ds3_test_block_q4_K));
    ds3_metal_buffer_t *C_buf   = ds3_metal_buffer_alloc(M * N * sizeof(float));
    CHECK(A_buf && W_buf && C_buf);

    CHECK(ds3_metal_buffer_write(A_buf, 0, A, M * K * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, N * nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);

    CHECK(ds3_metal_vec_matmul_q4k_simd(A_buf, W_buf, C_buf,
                                        K, N,
                                        nb_per_row * sizeof(ds3_test_block_q4_K)) == 0);
    CHECK(ds3_metal_buffer_read(C_buf, 0, C_vec, M * N * sizeof(float)) == 0);

    CHECK(ds3_metal_matmul_quantized_batch(A_buf, W_buf, C_buf,
                                            M, N, K,
                                            nb_per_row * sizeof(ds3_test_block_q4_K),
                                            0, DS3_TYPE_Q4_K) == 0);
    CHECK(ds3_metal_buffer_read(C_buf, 0, C_batch, M * N * sizeof(float)) == 0);

    float diff_vec   = max_diff(C_ref, C_vec, M * N);
    float diff_batch = max_diff(C_ref, C_batch, M * N);
    float diff_vb    = max_diff(C_vec, C_batch, M * N);
    CHECK(diff_vec   < 2e-4f);
    CHECK(diff_batch < 2e-4f);
    CHECK(diff_vb    < 2e-4f);

    ds3_metal_buffer_free(A_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(C_buf);
    free(A); free(W); free(C_ref); free(C_vec); free(C_batch);
    free(W_q); free(W_deq);

    fprintf(stderr, "PASS (diff ref=%.2e batch=%.2e vb=%.2e)\n",
            diff_vec, diff_batch, diff_vb);
    return 0;
}

static int test_batch_matmul_q6_k(void) {
    fprintf(stderr, "test_batch_matmul_q6_k ... ");

    const int M = 16;
    const int N = 64;
    const int K = 256;

    float *A = calloc(M * K, sizeof(float));
    float *W = calloc(N * K, sizeof(float));
    float *C_ref = calloc(M * N, sizeof(float));
    float *C_mtl = calloc(M * N, sizeof(float));
    CHECK(A && W && C_ref && C_mtl);

    srand(73);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < N * K; i++) W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    int nb_per_row = K / Q6K_BLOCK_SIZE;
    ds3_test_block_q6_K *W_q = calloc(N * nb_per_row, sizeof(ds3_test_block_q6_K));
    CHECK(W_q);
    for (int r = 0; r < N; r++) quantize_row_q6_K(W + r * K, W_q + r * nb_per_row, K);

    ds3_tensor_t W_tensor = {0};
    W_tensor.type = DS3_TYPE_Q6_K;
    W_tensor.n_dims = 2;
    W_tensor.ne[0] = K;
    W_tensor.ne[1] = N;
    W_tensor.data = W_q;
    W_tensor.size = N * nb_per_row * sizeof(ds3_test_block_q6_K);
    float *W_deq = ds3_ref_dequantize_tensor(&W_tensor);
    CHECK(W_deq);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * W_deq[n * K + k];
            C_ref[m * N + n] = sum;
        }
    }

    ds3_metal_buffer_t *A_buf = ds3_metal_buffer_alloc(M * K * sizeof(float));
    ds3_metal_buffer_t *W_buf = ds3_metal_buffer_alloc(N * nb_per_row * sizeof(ds3_test_block_q6_K));
    ds3_metal_buffer_t *C_buf = ds3_metal_buffer_alloc(M * N * sizeof(float));
    CHECK(A_buf && W_buf && C_buf);

    CHECK(ds3_metal_buffer_write(A_buf, 0, A, M * K * sizeof(float)) == 0);
    CHECK(ds3_metal_buffer_write(W_buf, 0, W_q, N * nb_per_row * sizeof(ds3_test_block_q6_K)) == 0);

    CHECK(ds3_metal_matmul_quantized_batch(A_buf, W_buf, C_buf,
                                            M, N, K,
                                            nb_per_row * sizeof(ds3_test_block_q6_K),
                                            0, DS3_TYPE_Q6_K) == 0);
    CHECK(ds3_metal_buffer_read(C_buf, 0, C_mtl, M * N * sizeof(float)) == 0);

    float diff = max_diff(C_ref, C_mtl, M * N);
    CHECK(diff < 3e-4f);

    ds3_metal_buffer_free(A_buf);
    ds3_metal_buffer_free(W_buf);
    ds3_metal_buffer_free(C_buf);
    free(A); free(W); free(C_ref); free(C_mtl); free(W_q); free(W_deq);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* ============================================================================ */

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    if (ds3_metal_init() != 0) {
        fprintf(stderr, "Metal init failed, skipping tests.\n");
        return 77;
    }

    int failed = 0;
    failed |= test_vec_matmul_f32();
    failed |= test_vec_matmul_f32_strided();
    failed |= test_vec_matmul_f32_simd();
    failed |= test_vec_matmul_q8_0();
    failed |= test_vec_matmul_q8_0_simd();
    failed |= test_vec_matmul_q4_k();
    failed |= test_vec_matmul_q4k_simd();
    failed |= test_vec_matmul_q6_k();
    failed |= test_vec_matmul_q6k_simd();
    failed |= test_vec_matmul_q4k_half();
    failed |= test_batch_matmul_q8_0();
    failed |= test_batch_matmul_q4_k();
    failed |= test_batch_vs_vec_q4_k();
    failed |= test_batch_matmul_q6_k();

    ds3_metal_shutdown();

    if (failed) {
        fprintf(stderr, "\n%d test(s) FAILED\n", failed);
        return 1;
    }
    fprintf(stderr, "\nAll quantized matmul tests PASSED\n");
    return 0;
}
