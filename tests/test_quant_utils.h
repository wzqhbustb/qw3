/*
 * test_quant_utils.h — Shared test helpers for quantization.
 */

#ifndef TEST_QUANT_UTILS_H
#define TEST_QUANT_UTILS_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* Local FP16 -> FP32 conversion for tests. */
static inline float test_fp16_to_fp32(uint16_t h)
{
#ifdef __FLT16_MAX__
    _Float16 f;
    memcpy(&f, &h, sizeof(f));
    return (float)f;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t u = sign; float f; memcpy(&f, &u, 4); return f; }
        int e = 1; while ((mant & 0x400) == 0) { mant <<= 1; e--; }
        mant &= 0x3FF; exp = e + 112;
    } else if (exp == 0x1F) { exp = 0xFF; } else { exp += 112; }
    uint32_t u = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &u, 4); return f;
#endif
}

/* ============================================================================
 * Q8_0
 * ============================================================================ */

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} ds3_test_block_q8_0;

static inline void quantize_row_q8_0(const float *src, ds3_test_block_q8_0 *dst, int n) {
    const int qk = 32;
    const int nb = n / qk;
    for (int b = 0; b < nb; b++) {
        float amax = 0.0f;
        for (int i = 0; i < qk; i++) {
            float v = fabsf(src[b * qk + i]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        if (d == 0.0f) d = 1.0f;

        _Float16 dh = (_Float16)d;
        memcpy(&dst[b].d, &dh, sizeof(uint16_t));

        for (int i = 0; i < qk; i++) {
            float v = src[b * qk + i] / d;
            int q = (int)roundf(v);
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[b].qs[i] = (int8_t)q;
        }
    }
}

/* ============================================================================
 * Q4_K
 * ============================================================================ */

#define Q4K_BLOCK_SIZE 256
#define Q4K_SCALE_SIZE 12

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[Q4K_SCALE_SIZE];
    uint8_t  qs[Q4K_BLOCK_SIZE / 2];
} ds3_test_block_q4_K;

static inline void encode_scale_min(int is, uint8_t sc, uint8_t m, uint8_t *scales) {
    if (is < 4) {
        scales[is]       = (scales[is]       & 0xC0) | (sc & 0x3F);
        scales[is + 4]   = (scales[is + 4]   & 0xC0) | (m  & 0x3F);
    } else {
        int k = is - 4;
        scales[k]        = (scales[k]        & 0x3F) | ((sc >> 4) << 6);
        scales[k + 4]    = (scales[k + 4]    & 0x3F) | ((m  >> 4) << 6);
        scales[is + 4]   = ((sc & 0x0F) << 0) | ((m & 0x0F) << 4);
    }
}

static inline void quantize_row_q4_K(const float *src, ds3_test_block_q4_K *dst, int n) {
    const int nb = n / Q4K_BLOCK_SIZE;
    for (int b = 0; b < nb; b++) {
        const float *x = src + b * Q4K_BLOCK_SIZE;

        float max_abs = 0.0f;
        float min_val = 0.0f;
        float max_val = 0.0f;
        for (int i = 0; i < Q4K_BLOCK_SIZE; i++) {
            if (x[i] < min_val) min_val = x[i];
            if (x[i] > max_val) max_val = x[i];
            float a = fabsf(x[i]);
            if (a > max_abs) max_abs = a;
        }

        float dmin = -min_val / 63.0f;
        if (dmin == 0.0f) dmin = 1.0f;
        float d = (max_val - min_val) / 15.0f / 63.0f;
        if (d == 0.0f) d = 1.0f;

        _Float16 dh = (_Float16)d;
        _Float16 dminh = (_Float16)dmin;
        memcpy(&dst[b].d, &dh, sizeof(uint16_t));
        memcpy(&dst[b].dmin, &dminh, sizeof(uint16_t));
        memset(dst[b].scales, 0, Q4K_SCALE_SIZE);

        int is = 0;
        for (int j = 0; j < Q4K_BLOCK_SIZE; j += 64) {
            for (int sub = 0; sub < 2; sub++) {
                int sub_j = j + sub * 32;
                float sub_min = x[sub_j];
                float sub_max = x[sub_j];
                for (int i = sub_j; i < sub_j + 32; i++) {
                    if (x[i] < sub_min) sub_min = x[i];
                    if (x[i] > sub_max) sub_max = x[i];
                }

                float sc_f = (sub_max - sub_min) / 15.0f / d;
                float m_f  = -sub_min / dmin;
                uint8_t sc = (uint8_t)roundf(sc_f);
                uint8_t m  = (uint8_t)roundf(m_f);
                if (sc > 63) sc = 63;
                if (m  > 63) m  = 63;

                encode_scale_min(is + sub, sc, m, dst[b].scales);

                float act_d = d * sc;
                float act_m = dmin * m;
                if (act_d == 0.0f) act_d = 1.0f;

                for (int l = 0; l < 32; l++) {
                    int qi = (int)roundf((x[sub_j + l] + act_m) / act_d);
                    if (qi < 0) qi = 0;
                    if (qi > 15) qi = 15;
                    if (sub == 0) {
                        dst[b].qs[(sub_j + l) / 2] = (dst[b].qs[(sub_j + l) / 2] & 0xF0) | (qi & 0x0F);
                    } else {
                        dst[b].qs[(sub_j + l) / 2] = (dst[b].qs[(sub_j + l) / 2] & 0x0F) | ((qi & 0x0F) << 4);
                    }
                }
            }
            is += 2;
        }
    }
}

/* ============================================================================
 * Q6_K
 * ============================================================================ */

#define Q6K_BLOCK_SIZE 256

typedef struct {
    uint8_t  ql[128];   /* lower 4 bits of 256 quants */
    uint8_t  qh[64];    /* upper 2 bits of 256 quants */
    int8_t   scales[16];/* per 16-weight group scales */
    uint16_t d;         /* super-block scale (FP16) */
} ds3_test_block_q6_K;

static inline void quantize_row_q6_K(const float *src, ds3_test_block_q6_K *dst, int n) {
    const int nb = n / Q6K_BLOCK_SIZE;
    for (int b = 0; b < nb; b++) {
        const float *x = src + b * Q6K_BLOCK_SIZE;

        float amax = 0.0f;
        for (int i = 0; i < Q6K_BLOCK_SIZE; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        float d = amax / (127.0f * 31.0f);
        if (d == 0.0f) d = 1.0f;

        _Float16 dh = (_Float16)d;
        memcpy(&dst[b].d, &dh, sizeof(uint16_t));
        memset(dst[b].ql, 0, sizeof(dst[b].ql));
        memset(dst[b].qh, 0, sizeof(dst[b].qh));

        for (int h = 0; h < 2; h++) {
            const int base = h * 128;
            const int sc_base = h * 8;
            for (int j = 0; j < 4; j++) {
                for (int is = 0; is < 2; is++) {
                    int sc_idx = sc_base + is + 2 * j;
                    float gmax = 0.0f;
                    for (int l = is * 16; l < (is + 1) * 16; l++) {
                        float a = fabsf(x[base + l + j * 32]);
                        if (a > gmax) gmax = a;
                    }
                    int scale = (int)roundf(gmax / d / 31.0f);
                    if (scale < 1) scale = 1;
                    if (scale > 127) scale = 127;
                    dst[b].scales[sc_idx] = (int8_t)scale;

                    for (int l = is * 16; l < (is + 1) * 16; l++) {
                        float v = x[base + l + j * 32];
                        int q = (int)roundf(v / d / (float)scale);
                        if (q < -32) q = -32;
                        if (q >  31) q =  31;
                        unsigned q_raw = (unsigned)(q + 32); /* 0..63 */
                        uint8_t lo = (uint8_t)(q_raw & 0x0F);
                        uint8_t hi = (uint8_t)((q_raw >> 4) & 0x03);
                        if (j == 0) {
                            dst[b].ql[l] = (dst[b].ql[l] & 0xF0) | lo;
                            dst[b].qh[l] = (uint8_t)((dst[b].qh[l] & ~0x03) | (hi << 0));
                        } else if (j == 1) {
                            dst[b].ql[l + 32] = (dst[b].ql[l + 32] & 0xF0) | lo;
                            dst[b].qh[l] = (uint8_t)((dst[b].qh[l] & ~0x0C) | (hi << 2));
                        } else if (j == 2) {
                            dst[b].ql[l] = (dst[b].ql[l] & 0x0F) | (lo << 4);
                            dst[b].qh[l] = (uint8_t)((dst[b].qh[l] & ~0x30) | (hi << 4));
                        } else { /* j == 3 */
                            dst[b].ql[l + 32] = (dst[b].ql[l + 32] & 0x0F) | (lo << 4);
                            dst[b].qh[l] = (uint8_t)((dst[b].qh[l] & ~0xC0) | (hi << 6));
                        }
                    }
                }
            }
        }
    }
}

#endif /* TEST_QUANT_UTILS_H */
