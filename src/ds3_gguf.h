/*
 * ds3_gguf.h — Minimal GGUF file format parser
 *
 * Implements GGUF v3 spec:
 *   https://github.com/ggml-org/gguf/blob/main/spec.md
 *
 * Design choices:
 *   - mmap-based: zero-copy, OS page cache manages eviction
 *   - No malloc for tensor metadata: pointers point directly into mmap
 *   - Minimal: only parses header, metadata keys, and tensor descriptors
 *   - Does NOT decompress/dequantize weights (that happens in Metal kernels)
 */

#ifndef DS3_GGUF_H
#define DS3_GGUF_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GGUF Format Constants
 * ============================================================================ */

#define DS3_GGUF_MAGIC      0x46554747  /* "GGUF" little-endian */
#define DS3_GGUF_VERSION    3

/* ggml_type enum values (subset — MUST match ds3.h ds3_type_t) */
typedef enum {
    DS3_GGUF_TYPE_F32      = 0,
    DS3_GGUF_TYPE_F16      = 1,
    DS3_GGUF_TYPE_Q4_0     = 2,
    DS3_GGUF_TYPE_Q5_0     = 6,
    DS3_GGUF_TYPE_Q8_0     = 8,
    DS3_GGUF_TYPE_Q2_K     = 10,
    DS3_GGUF_TYPE_Q3_K     = 11,
    DS3_GGUF_TYPE_Q4_K     = 12,
    DS3_GGUF_TYPE_Q5_K     = 13,
    DS3_GGUF_TYPE_Q6_K     = 14,
    DS3_GGUF_TYPE_Q8_K     = 15,
    DS3_GGUF_TYPE_IQ2_XXS  = 16,
    DS3_GGUF_TYPE_IQ2_XS   = 17,
    DS3_GGUF_TYPE_IQ3_XXS  = 18,
} ds3_gguf_type_t;

/* Metadata value type IDs */
typedef enum {
    DS3_GGUF_METADATA_TYPE_UINT8   = 0,
    DS3_GGUF_METADATA_TYPE_INT8    = 1,
    DS3_GGUF_METADATA_TYPE_UINT16  = 2,
    DS3_GGUF_METADATA_TYPE_INT16   = 3,
    DS3_GGUF_METADATA_TYPE_UINT32  = 4,
    DS3_GGUF_METADATA_TYPE_INT32   = 5,
    DS3_GGUF_METADATA_TYPE_FLOAT32 = 6,
    DS3_GGUF_METADATA_TYPE_BOOL    = 7,
    DS3_GGUF_METADATA_TYPE_STRING  = 8,
    DS3_GGUF_METADATA_TYPE_ARRAY   = 9,
    DS3_GGUF_METADATA_TYPE_UINT64  = 10,
    DS3_GGUF_METADATA_TYPE_INT64   = 11,
    DS3_GGUF_METADATA_TYPE_FLOAT64 = 12,
} ds3_gguf_metadata_type_t;

/* ============================================================================
 * Data Structures (point directly into mmap'd file)
 * ============================================================================ */

/* String: length + bytes (NOT null-terminated in file, but we null-terminate on load) */
typedef struct {
    uint64_t len;
    const char *data;   /* points into mmap; may NOT be null-terminated */
} ds3_gguf_str_t;

/* Metadata KV pair */
typedef struct {
    ds3_gguf_str_t key;
    uint32_t       value_type;  /* ds3_gguf_metadata_type_t */
    /* Value follows — interpreted based on value_type */
    union {
        uint8_t     u8;
        int8_t      i8;
        uint16_t    u16;
        int16_t     i16;
        uint32_t    u32;
        int32_t     i32;
        float       f32;
        bool        b;
        uint64_t    u64;
        int64_t     i64;
        double      f64;
        ds3_gguf_str_t str;
        struct {
            uint32_t type;      /* element type */
            uint64_t len;       /* number of elements */
            const void *data;   /* points into mmap */
        } array;
    } value;
} ds3_gguf_metadata_kv_t;

/* Tensor descriptor */
typedef struct {
    ds3_gguf_str_t name;
    uint32_t       n_dims;      /* 1..4 */
    uint64_t       ne[4];       /* dimensions (shape) */
    uint32_t       type;        /* ds3_gguf_type_t */
    uint64_t       offset;      /* offset from start of tensor data region */

    /* Computed after parsing */
    uint64_t       size;        /* total bytes in file for this tensor */
    const void    *data;        /* pointer into mmap (base + data_offset + offset) */
} ds3_gguf_tensor_info_t;

/* Parsed GGUF file handle */
typedef struct ds3_gguf {
    /* File backing */
    int             fd;         /* file descriptor */
    size_t          file_size;  /* total mmap'd bytes */
    void           *mmap_base;  /* base pointer from mmap() */

    /* Header */
    uint32_t        version;
    uint64_t        n_tensors;
    uint64_t        n_metadata;

    /* Alignment (default 32, overridden by metadata) */
    uint32_t        alignment;

    /* Offsets into mmap (pointers derived from these) */
    const uint8_t  *metadata_ptr;       /* start of metadata KV pairs */
    const uint8_t  *tensor_info_ptr;    /* start of tensor info array */
    const uint8_t  *tensor_data_ptr;    /* start of raw tensor data */
    uint64_t        tensor_data_offset; /* file offset of tensor data region */

    /* Parsed arrays (point into mmap) */
    ds3_gguf_metadata_kv_t  *metadata;  /* array[n_metadata] */
    ds3_gguf_tensor_info_t  *tensors;   /* array[n_tensors] */
} ds3_gguf_t;

/* ============================================================================
 * API
 * ============================================================================ */

/* Open & parse a GGUF file (mmap, read header, build tensor index) */
ds3_gguf_t *ds3_gguf_open(const char *path);

/* Close (munmap, free parser structs — does NOT free tensor data) */
void        ds3_gguf_close(ds3_gguf_t *gguf);

/* Find tensor by exact name. Returns NULL if not found. */
ds3_gguf_tensor_info_t *ds3_gguf_find_tensor(ds3_gguf_t *gguf, const char *name);

/* Print summary (header, tensor count, total weight bytes) */
void        ds3_gguf_print_summary(const ds3_gguf_t *gguf);

/* Print all tensors (name, shape, type, offset, size) */
void        ds3_gguf_print_tensors(const ds3_gguf_t *gguf);

/* Print all metadata keys */
void        ds3_gguf_print_metadata(const ds3_gguf_t *gguf);

/* Find metadata KV by key (returns NULL if not found) */
const ds3_gguf_metadata_kv_t *ds3_gguf_find_metadata(const ds3_gguf_t *gguf, const char *key);

/* Get metadata value by key (returns false if key not found) */
bool ds3_gguf_get_metadata_string(const ds3_gguf_t *gguf, const char *key, const char **out_str, size_t *out_len);
bool ds3_gguf_get_metadata_u32(const ds3_gguf_t *gguf, const char *key, uint32_t *out_val);
bool ds3_gguf_get_metadata_f32(const ds3_gguf_t *gguf, const char *key, float *out_val);

/* Read one GGUF string from a byte pointer; returns pointer after the string */
const uint8_t *ds3_gguf_read_str(const uint8_t *p, ds3_gguf_str_t *out);

#ifdef __cplusplus
}
#endif

#endif /* DS3_GGUF_H */
