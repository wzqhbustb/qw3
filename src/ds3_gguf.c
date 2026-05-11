/*
 * ds3_gguf.c — GGUF v3 parser implementation
 *
 * Strategy:
 *   1. mmap entire file
 *   2. Parse header → get tensor_count, metadata_count
 *   3. Skip metadata (just advance pointer, we index key positions)
 *   4. Parse tensor info array
 *   5. Compute tensor data base offset (header + metadata + tensor_info, aligned)
 *   6. Point each tensor's ->data into mmap
 *
 * No copying: all pointers point directly into mmap'd region.
 */

#include "ds3_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* ============================================================================
 * Endian helpers (GGUF is little-endian)
 * ============================================================================ */

static inline uint16_t ds3_le16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static inline uint32_t ds3_le32(const uint8_t *p) {
    return (uint32_t)ds3_le16(p) | ((uint32_t)ds3_le16(p+2) << 16);
}
static inline uint64_t ds3_le64(const uint8_t *p) {
    return (uint64_t)ds3_le32(p) | ((uint64_t)ds3_le32(p+4) << 32);
}
static inline float ds3_le_f32(const uint8_t *p) {
    union { uint32_t u; float f; } conv;
    conv.u = ds3_le32(p);
    return conv.f;
}

/* ============================================================================
 * String helpers
 * ============================================================================ */

static int ds3_gguf_strcmp(const ds3_gguf_str_t *s, const char *cstr) {
    size_t clen = strlen(cstr);
    if (s->len != clen) return (int)(s->len - clen);
    return memcmp(s->data, cstr, clen);
}

/* ============================================================================
 * Safe low-level parsing: read from pointer, advance, with bounds checking
 * ============================================================================
 *
 * All _safe variants take an `end` pointer (one past the last valid byte).
 * They return NULL if the read would exceed the buffer.
 */

static const uint8_t *read_u32_safe(const uint8_t *p, const uint8_t *end, uint32_t *out) {
    if (!p || p + 4 > end) return NULL;
    *out = ds3_le32(p);
    return p + 4;
}
static const uint8_t *read_u64_safe(const uint8_t *p, const uint8_t *end, uint64_t *out) {
    if (!p || p + 8 > end) return NULL;
    *out = ds3_le64(p);
    return p + 8;
}
static const uint8_t *read_str_safe(const uint8_t *p, const uint8_t *end, ds3_gguf_str_t *out) {
    if (!p) return NULL;
    p = read_u64_safe(p, end, &out->len);
    if (!p || p + out->len > end) return NULL;
    out->data = (const char *)p;
    return p + out->len;
}

/* Public API — fast path for tokenizer string-array iteration.
 * Caller (ds3_vocab_load) has already verified the array fits inside mmap. */
const uint8_t *ds3_gguf_read_str(const uint8_t *p, ds3_gguf_str_t *out) {
    out->len = ds3_le64(p);
    p += 8;
    out->data = (const char *)p;
    return p + out->len;
}

/* Read a metadata value based on its type ID. Returns pointer after value or NULL on overflow. */
static const uint8_t *read_metadata_value_safe(const uint8_t *p, const uint8_t *end, uint32_t type, ds3_gguf_metadata_kv_t *kv) {
    if (!p) return NULL;
    switch (type) {
        case DS3_GGUF_METADATA_TYPE_UINT8:
            if (p + 1 > end) return NULL;
            kv->value.u8  = *p++; break;
        case DS3_GGUF_METADATA_TYPE_INT8:
            if (p + 1 > end) return NULL;
            kv->value.i8  = (int8_t)*p++; break;
        case DS3_GGUF_METADATA_TYPE_UINT16:
            if (p + 2 > end) return NULL;
            kv->value.u16 = ds3_le16(p); p += 2; break;
        case DS3_GGUF_METADATA_TYPE_INT16:
            if (p + 2 > end) return NULL;
            kv->value.i16 = (int16_t)ds3_le16(p); p += 2; break;
        case DS3_GGUF_METADATA_TYPE_UINT32:
            p = read_u32_safe(p, end, &kv->value.u32); break;
        case DS3_GGUF_METADATA_TYPE_INT32:
            { uint32_t u; p = read_u32_safe(p, end, &u); if (p) kv->value.i32 = (int32_t)u; } break;
        case DS3_GGUF_METADATA_TYPE_FLOAT32:
            if (p + 4 > end) return NULL;
            kv->value.f32 = ds3_le_f32(p); p += 4; break;
        case DS3_GGUF_METADATA_TYPE_BOOL:
            if (p + 1 > end) return NULL;
            kv->value.b = (*p++ != 0); break;
        case DS3_GGUF_METADATA_TYPE_UINT64:
            p = read_u64_safe(p, end, &kv->value.u64); break;
        case DS3_GGUF_METADATA_TYPE_INT64:
            { uint64_t u; p = read_u64_safe(p, end, &u); if (p) kv->value.i64 = (int64_t)u; } break;
        case DS3_GGUF_METADATA_TYPE_FLOAT64:
            if (p + 8 > end) return NULL;
            { union { uint64_t u; double f; } conv;
              conv.u = ds3_le64(p); kv->value.f64 = conv.f; p += 8; }
            break;
        case DS3_GGUF_METADATA_TYPE_STRING:
            p = read_str_safe(p, end, &kv->value.str); break;
        case DS3_GGUF_METADATA_TYPE_ARRAY: {
            uint32_t arr_type;
            uint64_t arr_len;
            p = read_u32_safe(p, end, &arr_type);
            if (!p) return NULL;
            p = read_u64_safe(p, end, &arr_len);
            if (!p) return NULL;
            kv->value.array.type = arr_type;
            kv->value.array.len  = arr_len;
            kv->value.array.data = p;
            /* Skip array elements without deep-parsing */
            uint64_t elem_size = 0;
            switch (arr_type) {
                case DS3_GGUF_METADATA_TYPE_UINT8:
                case DS3_GGUF_METADATA_TYPE_INT8:    elem_size = 1; break;
                case DS3_GGUF_METADATA_TYPE_UINT16:
                case DS3_GGUF_METADATA_TYPE_INT16:   elem_size = 2; break;
                case DS3_GGUF_METADATA_TYPE_UINT32:
                case DS3_GGUF_METADATA_TYPE_INT32:
                case DS3_GGUF_METADATA_TYPE_FLOAT32: elem_size = 4; break;
                case DS3_GGUF_METADATA_TYPE_UINT64:
                case DS3_GGUF_METADATA_TYPE_INT64:
                case DS3_GGUF_METADATA_TYPE_FLOAT64: elem_size = 8; break;
                case DS3_GGUF_METADATA_TYPE_BOOL:    elem_size = 1; break;
                case DS3_GGUF_METADATA_TYPE_STRING: {
                    for (uint64_t i = 0; i < arr_len; i++) {
                        ds3_gguf_str_t s;
                        p = read_str_safe(p, end, &s);
                        if (!p) return NULL;
                    }
                    elem_size = 0;
                    break;
                }
                default: elem_size = 0; break;
            }
            if (elem_size > 0) {
                uint64_t skip = arr_len * elem_size;
                if (p + skip > end) return NULL;
                p += skip;
            }
            break;
        }
        default:
            fprintf(stderr, "[gguf] Unknown metadata value type: %u\n", type);
            break;
    }
    return p;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

ds3_gguf_t *ds3_gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[gguf] Cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("[gguf] fstat");
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void *mmap_base = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_base == MAP_FAILED) {
        perror("[gguf] mmap");
        close(fd);
        return NULL;
    }

    ds3_gguf_t *gguf = calloc(1, sizeof(ds3_gguf_t));
    if (!gguf) {
        munmap(mmap_base, file_size);
        close(fd);
        return NULL;
    }

    gguf->fd = fd;
    gguf->file_size = file_size;
    gguf->mmap_base = mmap_base;
    gguf->alignment = 32; /* default */

    /* --- Parse header --- */
    const uint8_t *p = (const uint8_t *)mmap_base;
    const uint8_t *end = p + file_size;

    if (file_size < 4) {
        fprintf(stderr, "[gguf] File too small for magic\n");
        goto fail;
    }
    uint32_t magic = ds3_le32(p);
    if (magic != DS3_GGUF_MAGIC) {
        fprintf(stderr, "[gguf] Bad magic: 0x%08X (expected 0x%08X)\n", magic, DS3_GGUF_MAGIC);
        goto fail;
    }
    p += 4;

    p = read_u32_safe(p, end, &gguf->version);
    if (!p || gguf->version < 3) {
        fprintf(stderr, "[gguf] Unsupported version %u (need v3)\n", gguf ? gguf->version : 0);
        goto fail;
    }

    p = read_u64_safe(p, end, &gguf->n_tensors);
    if (!p) goto fail;
    p = read_u64_safe(p, end, &gguf->n_metadata);
    if (!p) goto fail;

    /* --- Parse metadata KV pairs --- */
    if (gguf->n_metadata > SIZE_MAX / sizeof(ds3_gguf_metadata_kv_t)) {
        fprintf(stderr, "[gguf] n_metadata too large (%llu)\n", (unsigned long long)gguf->n_metadata);
        goto fail;
    }
    gguf->metadata = calloc((size_t)gguf->n_metadata, sizeof(ds3_gguf_metadata_kv_t));
    if (!gguf->metadata) goto fail;

    gguf->metadata_ptr = p;
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        ds3_gguf_metadata_kv_t *kv = &gguf->metadata[i];
        p = read_str_safe(p, end, &kv->key);
        if (!p) goto fail;
        p = read_u32_safe(p, end, &kv->value_type);
        if (!p) goto fail;
        p = read_metadata_value_safe(p, end, kv->value_type, kv);
        if (!p) goto fail;

        /* Check for alignment override */
        if (ds3_gguf_strcmp(&kv->key, "general.alignment") == 0 &&
            kv->value_type == DS3_GGUF_METADATA_TYPE_UINT32) {
            gguf->alignment = kv->value.u32;
        }
    }

    /* --- Parse tensor info --- */
    if (gguf->n_tensors > SIZE_MAX / sizeof(ds3_gguf_tensor_info_t)) {
        fprintf(stderr, "[gguf] n_tensors too large (%llu)\n", (unsigned long long)gguf->n_tensors);
        goto fail;
    }
    gguf->tensor_info_ptr = p;
    gguf->tensors = calloc((size_t)gguf->n_tensors, sizeof(ds3_gguf_tensor_info_t));
    if (!gguf->tensors) goto fail;

    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        ds3_gguf_tensor_info_t *ti = &gguf->tensors[i];
        p = read_str_safe(p, end, &ti->name);
        if (!p) goto fail;
        p = read_u32_safe(p, end, &ti->n_dims);
        if (!p) goto fail;
        if (ti->n_dims > 4) {
            fprintf(stderr, "[gguf] Tensor '%.*s' has %u dims (max 4)\n",
                    (int)ti->name.len, ti->name.data, ti->n_dims);
            goto fail;
        }
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            p = read_u64_safe(p, end, &ti->ne[d]);
            if (!p) goto fail;
        }
        p = read_u32_safe(p, end, &ti->type);
        if (!p) goto fail;
        p = read_u64_safe(p, end, &ti->offset);
        if (!p) goto fail;
    }

    /* --- Compute tensor data region start --- */
    uint64_t tensor_info_end = (uint64_t)(p - (const uint8_t *)mmap_base);
    uint64_t padding = (gguf->alignment - (tensor_info_end % gguf->alignment)) % gguf->alignment;
    gguf->tensor_data_offset = tensor_info_end + padding;
    gguf->tensor_data_ptr = (const uint8_t *)mmap_base + gguf->tensor_data_offset;

    /* --- Compute each tensor's data pointer and size --- */
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        ds3_gguf_tensor_info_t *ti = &gguf->tensors[i];
        ti->data = gguf->tensor_data_ptr + ti->offset;

        /* Compute element count */
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            n_elements *= ti->ne[d];
        }

        /* Compute bytes based on type */
        uint64_t type_size = 0;
        uint64_t block_size = 1;
        switch (ti->type) {
            case DS3_GGUF_TYPE_F32:  type_size = 4; block_size = 1; break;
            case DS3_GGUF_TYPE_F16:  type_size = 2; block_size = 1; break;
            case DS3_GGUF_TYPE_Q4_0: type_size = 18; block_size = 32; break; /* half(2) + qs[16] */
            case DS3_GGUF_TYPE_Q5_0: type_size = 22; block_size = 32; break;
            case DS3_GGUF_TYPE_Q8_0: type_size = 34; block_size = 32; break; /* half(2) + qs[32] */
            case DS3_GGUF_TYPE_Q2_K: type_size = 84; block_size = 256; break;
            case DS3_GGUF_TYPE_Q3_K: type_size = 110; block_size = 256; break;
            case DS3_GGUF_TYPE_Q4_K: type_size = 144; block_size = 256; break;
            case DS3_GGUF_TYPE_Q5_K: type_size = 176; block_size = 256; break;
            case DS3_GGUF_TYPE_Q6_K: type_size = 210; block_size = 256; break;
            case DS3_GGUF_TYPE_Q8_K: type_size = 290; block_size = 256; break; /* half(2) + qs[256] + bsums[32] */
            case DS3_GGUF_TYPE_IQ2_XXS: type_size = 66; block_size = 256; break;
            case DS3_GGUF_TYPE_IQ2_XS:  type_size = 74; block_size = 256; break;
            case DS3_GGUF_TYPE_IQ3_XXS: type_size = 98; block_size = 256; break;
            default:
                fprintf(stderr, "[gguf] Unknown tensor type %u for '%.*s'\n",
                        ti->type, (int)ti->name.len, ti->name.data);
                type_size = 0;
                break;
        }

        if (block_size > 1) {
            ti->size = (n_elements / block_size) * type_size;
            if (n_elements % block_size != 0) {
                ti->size += type_size; /* partial block */
            }
        } else {
            ti->size = n_elements * type_size;
        }
    }

    return gguf;

fail:
    ds3_gguf_close(gguf);
    return NULL;
}

void ds3_gguf_close(ds3_gguf_t *gguf) {
    if (!gguf) return;
    if (gguf->tensors) free(gguf->tensors);
    if (gguf->metadata) free(gguf->metadata);
    if (gguf->mmap_base && gguf->mmap_base != MAP_FAILED) {
        munmap(gguf->mmap_base, gguf->file_size);
    }
    if (gguf->fd >= 0) close(gguf->fd);
    free(gguf);
}

ds3_gguf_tensor_info_t *ds3_gguf_find_tensor(ds3_gguf_t *gguf, const char *name) {
    if (!gguf || !gguf->tensors) return NULL;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (ds3_gguf_strcmp(&gguf->tensors[i].name, name) == 0) {
            return &gguf->tensors[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Print / Debug
 * ============================================================================ */

static const char *gguf_type_name(uint32_t type) {
    switch (type) {
        case DS3_GGUF_TYPE_F32:      return "F32";
        case DS3_GGUF_TYPE_F16:      return "F16";
        case DS3_GGUF_TYPE_Q4_0:     return "Q4_0";
        case DS3_GGUF_TYPE_Q5_0:     return "Q5_0";
        case DS3_GGUF_TYPE_Q8_0:     return "Q8_0";
        case DS3_GGUF_TYPE_Q2_K:     return "Q2_K";
        case DS3_GGUF_TYPE_Q3_K:     return "Q3_K";
        case DS3_GGUF_TYPE_Q4_K:     return "Q4_K";
        case DS3_GGUF_TYPE_Q5_K:     return "Q5_K";
        case DS3_GGUF_TYPE_Q6_K:     return "Q6_K";
        case DS3_GGUF_TYPE_Q8_K:     return "Q8_K";
        case DS3_GGUF_TYPE_IQ2_XXS:  return "IQ2_XXS";
        case DS3_GGUF_TYPE_IQ2_XS:   return "IQ2_XS";
        case DS3_GGUF_TYPE_IQ3_XXS:  return "IQ3_XXS";
        default:                     return "???";
    }
}

void ds3_gguf_print_summary(const ds3_gguf_t *gguf) {
    if (!gguf) return;
    printf("═══ GGUF Summary ═══\n");
    printf("Version:      %u\n", gguf->version);
    printf("Tensors:      %llu\n", (unsigned long long)gguf->n_tensors);
    printf("Metadata KVs: %llu\n", (unsigned long long)gguf->n_metadata);
    printf("Alignment:    %u\n", gguf->alignment);
    printf("Data offset:  0x%llX\n", (unsigned long long)gguf->tensor_data_offset);

    uint64_t total_weight_bytes = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        total_weight_bytes += gguf->tensors[i].size;
    }
    printf("Weight bytes: %.2f MB (%.2f GB)\n",
           total_weight_bytes / (1024.0 * 1024.0),
           total_weight_bytes / (1024.0 * 1024.0 * 1024.0));
}

void ds3_gguf_print_tensors(const ds3_gguf_t *gguf) {
    if (!gguf) return;
    printf("\n═══ Tensors (%llu) ═══\n", (unsigned long long)gguf->n_tensors);
    printf("%-50s %6s  %-8s  %-20s  %10s  %12s\n",
           "Name", "Dims", "Type", "Shape", "Offset", "Size");
    printf("%s\n", "─────────────────────────────────────────────────────────────────────────────────────────");

    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        const ds3_gguf_tensor_info_t *t = &gguf->tensors[i];
        char shape_str[64] = {0};
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d > 0) strncat(shape_str, ",", sizeof(shape_str) - strlen(shape_str) - 1);
            char dim_buf[32];
            snprintf(dim_buf, sizeof(dim_buf), "%llu", (unsigned long long)t->ne[d]);
            strncat(shape_str, dim_buf, sizeof(shape_str) - strlen(shape_str) - 1);
        }

        printf("%-50.*s %6u  %-8s  %-20s  0x%08llX  %10llu\n",
               (int)t->name.len, t->name.data,
               t->n_dims,
               gguf_type_name(t->type),
               shape_str,
               (unsigned long long)t->offset,
               (unsigned long long)t->size);
    }
}

void ds3_gguf_print_metadata(const ds3_gguf_t *gguf) {
    if (!gguf) return;
    printf("\n═══ Metadata (%llu) ═══\n", (unsigned long long)gguf->n_metadata);
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        const ds3_gguf_metadata_kv_t *kv = &gguf->metadata[i];
        printf("  %.*s = ", (int)kv->key.len, kv->key.data);
        switch (kv->value_type) {
            case DS3_GGUF_METADATA_TYPE_UINT32:  printf("%u\n", kv->value.u32); break;
            case DS3_GGUF_METADATA_TYPE_INT32:   printf("%d\n", kv->value.i32); break;
            case DS3_GGUF_METADATA_TYPE_FLOAT32: printf("%g\n", kv->value.f32); break;
            case DS3_GGUF_METADATA_TYPE_BOOL:    printf("%s\n", kv->value.b ? "true" : "false"); break;
            case DS3_GGUF_METADATA_TYPE_STRING:  printf("%.*s\n", (int)kv->value.str.len, kv->value.str.data); break;
            case DS3_GGUF_METADATA_TYPE_UINT64:  printf("%llu\n", (unsigned long long)kv->value.u64); break;
            case DS3_GGUF_METADATA_TYPE_INT64:   printf("%lld\n", (long long)kv->value.i64); break;
            case DS3_GGUF_METADATA_TYPE_ARRAY:   printf("[array, len=%llu]\n", (unsigned long long)kv->value.array.len); break;
            default: printf("(type=%u)\n", kv->value_type); break;
        }
    }
}

/* ============================================================================
 * Metadata getters
 * ============================================================================ */

const ds3_gguf_metadata_kv_t *ds3_gguf_find_metadata(const ds3_gguf_t *gguf, const char *key) {
    if (!gguf || !gguf->metadata) return NULL;
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        if (ds3_gguf_strcmp(&gguf->metadata[i].key, key) == 0) {
            return &gguf->metadata[i];
        }
    }
    return NULL;
}

bool ds3_gguf_get_metadata_string(const ds3_gguf_t *gguf, const char *key, const char **out_str, size_t *out_len) {
    if (!gguf || !gguf->metadata) return false;
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        if (ds3_gguf_strcmp(&gguf->metadata[i].key, key) == 0 &&
            gguf->metadata[i].value_type == DS3_GGUF_METADATA_TYPE_STRING) {
            *out_str = gguf->metadata[i].value.str.data;
            *out_len = gguf->metadata[i].value.str.len;
            return true;
        }
    }
    return false;
}

bool ds3_gguf_get_metadata_u32(const ds3_gguf_t *gguf, const char *key, uint32_t *out_val) {
    if (!gguf || !gguf->metadata) return false;
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        if (ds3_gguf_strcmp(&gguf->metadata[i].key, key) == 0 &&
            gguf->metadata[i].value_type == DS3_GGUF_METADATA_TYPE_UINT32) {
            *out_val = gguf->metadata[i].value.u32;
            return true;
        }
    }
    return false;
}

bool ds3_gguf_get_metadata_f32(const ds3_gguf_t *gguf, const char *key, float *out_val) {
    if (!gguf || !gguf->metadata) return false;
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        if (ds3_gguf_strcmp(&gguf->metadata[i].key, key) == 0 &&
            gguf->metadata[i].value_type == DS3_GGUF_METADATA_TYPE_FLOAT32) {
            *out_val = gguf->metadata[i].value.f32;
            return true;
        }
    }
    return false;
}
