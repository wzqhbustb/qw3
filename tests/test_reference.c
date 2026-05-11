/*
 * test_reference.c — Unit tests for CPU reference operators
 *
 * Build: cc -O0 -g -I.. ../src/ds3_reference.c test_reference.c -o test_reference -lm
 * Run:   ./test_reference
 */

#include "../src/ds3_reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-4f

static int g_pass = 0;
static int g_fail = 0;

static void check(const char *name, float got, float expected)
{
    float diff = fabsf(got - expected);
    if (diff <= EPS) {
        g_pass++;
    } else {
        g_fail++;
        printf("FAIL %s: got %.6f expected %.6f (diff %.6f)\n",
               name, got, expected, diff);
    }
}

static void test_rms_norm(void)
{
    printf("test_rms_norm ... ");
    int prev = g_pass + g_fail;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    ds3_ref_rms_norm(x, w, 1e-6f, 4, out);
    /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386 */
    float rms = sqrtf(7.5f);
    check("rms_norm[0]", out[0], 1.0f / rms);
    check("rms_norm[1]", out[1], 2.0f / rms);
    check("rms_norm[2]", out[2], 3.0f / rms);
    check("rms_norm[3]", out[3], 4.0f / rms);

    /* weight != 1 */
    float w2[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    ds3_ref_rms_norm(x, w2, 1e-6f, 4, out);
    check("rms_norm_w2[0]", out[0], 2.0f / rms);
    check("rms_norm_w2[3]", out[3], 8.0f / rms);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_softmax(void)
{
    printf("test_softmax ... ");
    int prev = g_pass + g_fail;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ds3_ref_softmax(x, 4);
    float sum = x[0] + x[1] + x[2] + x[3];
    check("softmax_sum", sum, 1.0f);
    check("softmax[0]",  x[0], 0.0320586f);  /* e^1 / sum(e^i) */
    check("softmax[3]",  x[3], 0.6439143f);  /* e^4 / sum(e^i) */

    /* all equal → uniform */
    float y[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    ds3_ref_softmax(y, 4);
    check("softmax_uniform", y[0], 0.25f);
    check("softmax_uniform_sum", y[0]+y[1]+y[2]+y[3], 1.0f);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_matmul(void)
{
    printf("test_matmul ... ");
    int prev = g_pass + g_fail;
    /* A[2][3] = {{1,2,3},{4,5,6}} */
    float A[6] = {1,2,3,4,5,6};
    /* B[3][2] = {{1,0},{0,1},{1,1}} */
    float B[6] = {1,0,0,1,1,1};
    float C[4];
    ds3_ref_matmul(A, B, C, 2, 2, 3);
    /* C[0][0] = 1*1 + 2*0 + 3*1 = 4 */
    /* C[0][1] = 1*0 + 2*1 + 3*1 = 5 */
    /* C[1][0] = 4*1 + 5*0 + 6*1 = 10 */
    /* C[1][1] = 4*0 + 5*1 + 6*1 = 11 */
    check("matmul[0,0]", C[0], 4.0f);
    check("matmul[0,1]", C[1], 5.0f);
    check("matmul[1,0]", C[2], 10.0f);
    check("matmul[1,1]", C[3], 11.0f);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_silu(void)
{
    printf("test_silu ... ");
    int prev = g_pass + g_fail;
    float x[3] = {0.0f, 1.0f, -1.0f};
    float out[3];
    ds3_ref_silu(x, out, 3);
    check("silu[0]", out[0], 0.0f);
    check("silu[1]", out[1], 1.0f / (1.0f + expf(-1.0f))); /* x * sigmoid(x) */
    check("silu[2]", out[2], -1.0f / (1.0f + expf(1.0f)));
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_rope(void)
{
    printf("test_rope ... ");
    int prev = g_pass + g_fail;
    /* Qwen3 uses GPT-NeoX style RoPE: pairs are (i, i + head_dim/2).
     * Single head, head_dim = 4, seq_pos = 1.
     * Pair 0: indices (0, 2), freq = theta^0 = 1.
     * Pair 1: indices (1, 3), freq = theta^(-2/4) = 1e-3, angle = 1e-3 rad. */
    float q[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    float k[4] = {1.0f, 0.0f, 1.0f, 0.0f};

    float q_norm_before = 0.0f;
    for (int i = 0; i < 4; i++) q_norm_before += q[i] * q[i];

    ds3_ref_rope(q, k, 0, 1, 1, 4);

    float q_norm_after = 0.0f;
    for (int i = 0; i < 4; i++) q_norm_after += q[i] * q[i];

    /* RoPE is a rotation matrix → preserves L2 norm */
    check("rope_norm_preserve", q_norm_after, q_norm_before);

    /* At seq_pos=0, cos=1, sin=0 → no change */
    check("rope_q[0]@pos0", q[0], 1.0f);
    check("rope_q[1]@pos0", q[1], 0.0f);

    /* Now test seq_pos=1 */
    float q2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k2[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    ds3_ref_rope(q2, k2, 1, 1, 1, 4);

    /* Pair 0 (indices 0, 2): angle = 1 rad, (1, 0) -> (cos, sin) */
    check("rope_q2[0]@pos1", q2[0], cosf(1.0f));
    check("rope_q2[2]@pos1", q2[2], sinf(1.0f));
    /* Pair 1 (indices 1, 3): angle = 1e-3 rad, (0, 1) -> (-sin, cos) */
    check("rope_q2[1]@pos1", q2[1], -sinf(1e-3f));
    check("rope_q2[3]@pos1", q2[3], cosf(1e-3f));

    /* Multi-head test: 2 Q heads, 1 KV head */
    float q3[8] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f}; /* 2 heads */
    float k3[4] = {1.0f, 0.0f, 0.0f, 1.0f}; /* 1 head */
    ds3_ref_rope(q3, k3, 1, 2, 1, 4);
    /* head 0: same as q2 */
    check("rope_multi_q0[0]", q3[0], cosf(1.0f));
    check("rope_multi_q0[1]", q3[1], -sinf(1e-3f));
    /* head 1: (2,0,0,2) scaled */
    check("rope_multi_q1[0]", q3[4], 2.0f * cosf(1.0f));
    check("rope_multi_q1[1]", q3[5], -2.0f * sinf(1e-3f));
    /* Pair 1 angle ≈ 0 → index 3 stays ≈ 2 */
    check("rope_multi_q1[3]", q3[7], 2.0f);

    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_topk(void)
{
    printf("test_topk ... ");
    int prev = g_pass + g_fail;
    float scores[8] = {0.1f, 0.5f, 0.2f, 0.9f, 0.3f, 0.8f, 0.0f, 0.4f};
    int indices[3];
    float values[3];
    ds3_ref_topk(scores, 8, 3, indices, values);

    check("topk_idx[0]", (float)indices[0], 3.0f); /* 0.9 */
    check("topk_idx[1]", (float)indices[1], 5.0f); /* 0.8 */
    check("topk_idx[2]", (float)indices[2], 1.0f); /* 0.5 */
    check("topk_val[0]", values[0], 0.9f);
    check("topk_val[1]", values[1], 0.8f);
    check("topk_val[2]", values[2], 0.5f);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

/* -------------------------------------------------------------------------- */
/* Dequantization tests (local block defs — must match ds3_reference.c)       */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} test_block_q4_K;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(test_block_q4_K) == 144, "test_block_q4_K must match ds3_reference.c");
#endif

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} test_block_q8_0;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(test_block_q8_0) == 34, "test_block_q8_0 must match ds3_reference.c");
#endif

typedef struct {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
} test_block_q6_K;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(test_block_q6_K) == 210, "test_block_q6_K must match ds3_reference.c");
#endif

static void test_dequantize_q4_K(void)
{
    printf("test_dequantize_q4_K ... ");
    int prev = g_pass + g_fail;

    /* Build a single Q4_K block with known values.
     * d = 1.0, dmin = 0.0
     * First sub-block scale = 1, min = 0
     * qs[0] = 0xAB → w0 = A=10, w32 = B=11
     */
    test_block_q4_K block;
    memset(&block, 0, sizeof(block));

    /* d = 1.0 in FP16 = 0x3C00 */
    block.d = 0x3C00;
    /* dmin = 1.0 in FP16 = 0x3C00 (must be non-zero to test m1 > 0) */
    block.dmin = 0x3C00;

    /* scales[0] & 63 = 1 → first scale (d1) = 1 */
    block.scales[0] = 1;
    /* scales[1] & 63 = 1 → second scale (d2) = 1 */
    block.scales[1] = 1;
    /* scales[4] & 63 = 0 → first min (m1) = 0 */
    block.scales[4] = 0;
    /* scales[5] & 63 = 0 → second min (m2) = 0 */
    block.scales[5] = 0;

    /* 0xAB: low nibble = 0xB = 11, high nibble = 0xA = 10 */
    block.qs[0] = 0xAB;

    /* Test j >= 4 path (third 64-group, weights 128..191).
     * get_scale_min_k4(j=4) uses scales[8] low nibble + scales[0] bits 6-7.
     * get_scale_min_k4(j=5) uses scales[9] low nibble + scales[1] bits 6-7.
     *
     * Set scales[8] = 0x53 → low nibble = 3, high nibble = 5
     *   j=4: d = 3 | 0 = 3,   m = 5 | 0 = 5
     * Set scales[9] = 0x02 → low nibble = 2, high nibble = 0
     *   j=5: d = 2 | 0 = 2,   m = 0 | 0 = 0
     *
     * qs[64] = 0xCD → low nibble = 13, high nibble = 12
     * w128 = d1*13 - m1 = 3*13 - 5 = 34
     * w160 = d2*12 - m2 = 2*12 - 0 = 24
     */
    block.qs[64] = 0xCD;
    block.scales[8] = 0x53;  /* j=4: d=3, m=5  (j>=4 path) */
    block.scales[9] = 0x02;  /* j=5: d=2, m=0  (j>=4 path) */

    ds3_tensor_t t;
    memset(&t, 0, sizeof(t));
    t.type = DS3_TYPE_Q4_K;
    t.n_dims = 1;
    t.ne[0] = 256;
    t.data = &block;
    t.size = sizeof(block);

    float *out = ds3_ref_dequantize_tensor(&t);
    assert(out != NULL);

    /* --- First 64-group (j=0,1 < 4 path) --- */
    /* First loop:  low nibbles of qs[0..31] → w0..w31  using d1 */
    /* w0 = d1 * (qs[0] & 0xF) = 1.0 * 11 = 11.0 */
    check("q4k_w0",  out[0],  11.0f);

    /* Second loop: high nibbles of qs[0..31] → w32..w63 using d2 */
    /* w32 = d2 * (qs[0] >> 4) = 1.0 * 10 = 10.0 */
    check("q4k_w32", out[32], 10.0f);

    /* qs[1..31] are zero → remaining weights in this 64-group are 0 */
    check("q4k_w1",  out[1],   0.0f);
    check("q4k_w33", out[33],  0.0f);

    /* --- Third 64-group (j=4,5 >= 4 path) --- */
    /* j=4 (d1=3, m1=5): low nibbles of qs[64..95] → w128..w159 */
    check("q4k_w128", out[128], 34.0f);   /* 3*13 - 5 */

    /* j=5 (d2=2, m2=0): high nibbles of qs[64..95] → w160..w191 */
    check("q4k_w160", out[160], 24.0f);   /* 2*12 - 0 */

    /* qs[65..95] are zero → remaining weights in this group are -m1 or 0 */
    check("q4k_w129", out[129], -5.0f);   /* 3*0 - 5 */
    check("q4k_w161", out[161],  0.0f);   /* 2*0 - 0 */

    free(out);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_dequantize_q8_0(void)
{
    printf("test_dequantize_q8_0 ... ");
    int prev = g_pass + g_fail;

    test_block_q8_0 block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;      /* 1.0 in FP16 */
    block.qs[0] = 5;
    block.qs[1] = -3;

    ds3_tensor_t t;
    memset(&t, 0, sizeof(t));
    t.type = DS3_TYPE_Q8_0;
    t.n_dims = 1;
    t.ne[0] = 32;
    t.data = &block;
    t.size = sizeof(block);

    float *out = ds3_ref_dequantize_tensor(&t);
    assert(out != NULL);

    check("q8k_w0", out[0],  5.0f);
    check("q8k_w1", out[1], -3.0f);
    check("q8k_w2", out[2],  0.0f);

    free(out);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_dequantize_q6_K(void)
{
    printf("test_dequantize_q6_K ... ");
    int prev = g_pass + g_fail;

    test_block_q6_K block;
    memset(&block, 0, sizeof(block));

    /* d = 1.0 in FP16 = 0x3C00 */
    block.d = 0x3C00;
    /* All scales = 1 */
    for (int i = 0; i < 16; i++) block.scales[i] = 1;

    /* ql[0] = 0x31 → low nibble = 1, high nibble = 3 */
    block.ql[0] = 0x31;
    /* ql[32] = 0x42 → low nibble = 2, high nibble = 4 */
    block.ql[32] = 0x42;
    /* qh[0] = 0xAA → each 2-bit field = 2 (binary 10) */
    block.qh[0] = 0xAA;

    /* For l=0, is=0:
     * q1 = (ql[0]&0x0F | ((qh[0]>>0)&3)<<4) - 32
     *    = (1 | (2<<4)) - 32 = (1|32) - 32 = 1
     * q2 = (ql[32]&0x0F | ((qh[0]>>2)&3)<<4) - 32
     *    = (2 | (2<<4)) - 32 = (2|32) - 32 = 2
     * q3 = (ql[0]>>4 | ((qh[0]>>4)&3)<<4) - 32
     *    = (3 | (2<<4)) - 32 = (3|32) - 32 = 3
     * q4 = (ql[32]>>4 | ((qh[0]>>6)&3)<<4) - 32
     *    = (4 | (2<<4)) - 32 = (4|32) - 32 = 4
     *
     * y[0]   = d * sc[0] * q1 = 1 * 1 * 1 = 1
     * y[32]  = d * sc[2] * q2 = 1 * 1 * 2 = 2
     * y[64]  = d * sc[4] * q3 = 1 * 1 * 3 = 3
     * y[96]  = d * sc[6] * q4 = 1 * 1 * 4 = 4
     */

    ds3_tensor_t t;
    memset(&t, 0, sizeof(t));
    t.type = DS3_TYPE_Q6_K;
    t.n_dims = 1;
    t.ne[0] = 256;
    t.data = &block;
    t.size = sizeof(block);

    float *out = ds3_ref_dequantize_tensor(&t);
    assert(out != NULL);

    check("q6k_w0",  out[0],   1.0f);
    check("q6k_w32", out[32],  2.0f);
    check("q6k_w64", out[64],  3.0f);
    check("q6k_w96", out[96],  4.0f);

    /* All other weights should be zero (ql/qh zero → q = -32 + 0 = -32,
     * but scales are 1 and d=1, so weight = -32. Wait — if ql=0 and qh=0,
     * then q = (0 | 0) - 32 = -32. So y = -32.
     * Let's verify a zero position: l=1, ql[1]=0, ql[33]=0, qh[1]=0.
     * q1 = (0|0)-32 = -32. y[1] = -32.
     * But that's because we memset to zero. Let's just verify the known ones. */

    free(out);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_dequantize_f16(void)
{
    printf("test_dequantize_f16 ... ");
    int prev = g_pass + g_fail;

    uint16_t data[3] = {0x3C00, 0x4000, 0x0000}; /* 1.0, 2.0, 0.0 */

    ds3_tensor_t t;
    memset(&t, 0, sizeof(t));
    t.type = DS3_TYPE_F16;
    t.n_dims = 1;
    t.ne[0] = 3;
    t.data = data;
    t.size = sizeof(data);

    float *out = ds3_ref_dequantize_tensor(&t);
    assert(out != NULL);

    check("f16_0", out[0], 1.0f);
    check("f16_1", out[1], 2.0f);
    check("f16_2", out[2], 0.0f);

    free(out);
    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

static void test_dequantize_row(void)
{
    printf("test_dequantize_row ... ");
    int prev = g_pass + g_fail;

    /* F32: 2 rows of 3 elements each. */
    float f32_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    ds3_tensor_t t;
    memset(&t, 0, sizeof(t));
    t.type = DS3_TYPE_F32;
    t.n_dims = 2;
    t.ne[0] = 3;
    t.ne[1] = 2;
    t.nb[0] = sizeof(float);
    t.nb[1] = 3 * sizeof(float);
    t.data = f32_data;
    t.size = sizeof(f32_data);

    float row[3];
    assert(ds3_ref_dequantize_row(&t, 1, row) == 0);
    check("row_f32_0", row[0], 4.0f);
    check("row_f32_1", row[1], 5.0f);
    check("row_f32_2", row[2], 6.0f);

    /* F16: 2 rows of 3 elements. */
    uint16_t f16_data[6] = {0x3C00, 0x4000, 0x0000,  /* 1.0, 2.0, 0.0 */
                            0x4200, 0x4400, 0x4500}; /* 3.0, 4.0, 5.0 */
    memset(&t, 0, sizeof(t));
    t.type = DS3_TYPE_F16;
    t.n_dims = 2;
    t.ne[0] = 3;
    t.ne[1] = 2;
    t.nb[0] = sizeof(uint16_t);
    t.nb[1] = 3 * sizeof(uint16_t);
    t.data = f16_data;
    t.size = sizeof(f16_data);

    assert(ds3_ref_dequantize_row(&t, 0, row) == 0);
    check("row_f16_0", row[0], 1.0f);
    check("row_f16_1", row[1], 2.0f);
    check("row_f16_2", row[2], 0.0f);

    assert(ds3_ref_dequantize_row(&t, 1, row) == 0);
    check("row_f16_3", row[0], 3.0f);
    check("row_f16_4", row[1], 4.0f);
    check("row_f16_5", row[2], 5.0f);

    printf("done (%d new checks)\n", (g_pass + g_fail) - prev);
}

int main(void)
{
    printf("=== DS3 Reference Operator Tests ===\n\n");

    test_rms_norm();
    test_softmax();
    test_matmul();
    test_silu();
    test_rope();
    test_topk();
    test_dequantize_q4_K();
    test_dequantize_q8_0();
    test_dequantize_q6_K();
    test_dequantize_f16();
    test_dequantize_row();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
