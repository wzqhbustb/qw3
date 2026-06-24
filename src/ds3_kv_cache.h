/*
 * ds3 KV Cache provider interface.
 *
 * This header defines the C-side binary layout and provider abstraction
 * used by ds3_engine to interface with the Rust ds3-kv-cache-svc.
 *
 * Byte layout must match the Rust definitions in
 * /Users/wangyang/kv_cache/src/layout.rs.
 */

#ifndef DS3_KV_CACHE_H
#define DS3_KV_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Region header magic: "DS3KVC02" as little-endian uint64 */
#define DS3_KVC_REGION_MAGIC 0x323043564B335344ULL

/* Block header magic: "KVCK" padded with zeros */
#define DS3_KVC_BLOCK_MAGIC  0x000000004B43564BULL

/* Region format version */
#define DS3_KVC_VERSION      2

/* Region header size: one page */
#define DS3_KVC_REGION_HEADER_SIZE 4096

/* Block header size and alignment */
#define DS3_KVC_BLOCK_HEADER_SIZE  128
#define DS3_KVC_BLOCK_HEADER_ALIGN 128

/* Default model parameters for Qwen3-30B-A3B */
#define DS3_KVC_DEFAULT_N_LAYER   48
#define DS3_KVC_DEFAULT_N_KV_HEAD 4
#define DS3_KVC_DEFAULT_HEAD_DIM  128
#define DS3_KVC_DEFAULT_DTYPE_SIZE 2  /* FP16 */

/* Block lifecycle states. Must match BlockState in Rust. */
typedef enum {
    DS3_KVC_BLOCK_FREE = 0,
    DS3_KVC_BLOCK_RESERVED = 1,
    DS3_KVC_BLOCK_VALID = 2,
    DS3_KVC_BLOCK_STALE = 3,
} ds3_kv_block_state_t;

/* Region header, page-aligned. */
typedef struct __attribute__((aligned(4096))) {
    uint64_t magic;
    uint32_t version;
    uint32_t block_size;       /* tokens per block */
    uint64_t num_blocks;
    uint64_t block_data_size;  /* kv data bytes per block */
    uint32_t n_layer;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint64_t created_at;
    uint64_t checksum;
    uint8_t  reserved[4016];
} ds3_kv_region_header_t;

/* Block header, 128-byte aligned. */
typedef struct __attribute__((aligned(128))) {
    uint64_t magic;
    uint32_t state;
    int32_t  ref_count;
    uint64_t token_hash;
    uint16_t n_tokens;
    uint16_t padding_1;
    uint32_t padding_2;
    uint64_t block_id;
    char     session_owner[48];
    uint64_t reserved[4];
} ds3_kv_block_header_t;

/* Handle to an allocated block. */
typedef struct {
    uint64_t block_id;
    uint64_t mmap_offset;  /* offset of kv_data region within mmap */
} ds3_kv_block_handle_t;

/* Forward declaration. */
typedef struct ds3_kv_cache_provider ds3_kv_cache_provider_t;

/* Provider vtable. */
struct ds3_kv_cache_provider_vtable {
    const char *name;

    /* Open/close the provider. config is provider-specific. */
    ds3_kv_cache_provider_t *(*open)(const char *config);
    void (*close)(ds3_kv_cache_provider_t *p);

    /* Query the longest cached prefix length for tokens. */
    int (*lookup)(ds3_kv_cache_provider_t *p,
                  const char *session_id,
                  const int *tokens, int n_tokens,
                  int *out_cached_len);

    /* Read cached KV into kv_buffer (host memory). */
    int (*read)(ds3_kv_cache_provider_t *p,
                const char *session_id,
                int start_pos, int n_tokens,
                void *kv_buffer);

    /* Write KV back. tokens/n_tokens is the full sequence.
     * The provider internally skips already-cached prefix. */
    int (*write)(ds3_kv_cache_provider_t *p,
                 const char *session_id,
                 const int *tokens, int n_tokens,
                 const void *kv_buffer, size_t kv_buffer_bytes);

    /* Session lifecycle. */
    int (*create_session)(ds3_kv_cache_provider_t *p, const char *session_id);
    int (*close_session)(ds3_kv_cache_provider_t *p, const char *session_id);
    int (*reset_session)(ds3_kv_cache_provider_t *p, const char *session_id);
};

/* Provider instance wraps the vtable + provider-specific state. */
struct ds3_kv_cache_provider {
    const struct ds3_kv_cache_provider_vtable *vtable;
    void *priv;
};

/* Convenience wrappers. */
static inline ds3_kv_cache_provider_t *
ds3_kv_cache_provider_open(const struct ds3_kv_cache_provider_vtable *vtable, const char *config)
{
    return vtable->open(config);
}

static inline void ds3_kv_cache_provider_close(ds3_kv_cache_provider_t *p)
{
    if (p && p->vtable->close) p->vtable->close(p);
}

static inline int ds3_kv_cache_lookup(ds3_kv_cache_provider_t *p,
                                      const char *session_id,
                                      const int *tokens, int n_tokens,
                                      int *out_cached_len)
{
    if (!p) return -1;
    return p->vtable->lookup(p, session_id, tokens, n_tokens, out_cached_len);
}

static inline int ds3_kv_cache_read(ds3_kv_cache_provider_t *p,
                                    const char *session_id,
                                    int start_pos, int n_tokens,
                                    void *kv_buffer)
{
    if (!p) return -1;
    return p->vtable->read(p, session_id, start_pos, n_tokens, kv_buffer);
}

static inline int ds3_kv_cache_write(ds3_kv_cache_provider_t *p,
                                     const char *session_id,
                                     const int *tokens, int n_tokens,
                                     const void *kv_buffer, size_t kv_buffer_bytes)
{
    if (!p) return -1;
    return p->vtable->write(p, session_id, tokens, n_tokens, kv_buffer, kv_buffer_bytes);
}

static inline int ds3_kv_cache_create_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    if (!p) return -1;
    return p->vtable->create_session(p, session_id);
}

static inline int ds3_kv_cache_close_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    if (!p) return -1;
    return p->vtable->close_session(p, session_id);
}

static inline int ds3_kv_cache_reset_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    if (!p) return -1;
    return p->vtable->reset_session(p, session_id);
}

/* Built-in providers. */
extern const struct ds3_kv_cache_provider_vtable ds3_kv_local_provider;
extern const struct ds3_kv_cache_provider_vtable ds3_kv_service_provider;
extern const struct ds3_kv_cache_provider_vtable ds3_kv_fallback_provider;

/* Helpers */
static inline uint64_t ds3_kv_per_token_bytes(uint32_t n_layer,
                                              uint32_t n_kv_head,
                                              uint32_t head_dim,
                                              uint32_t dtype_size)
{
    return (uint64_t)n_layer * 2 * n_kv_head * head_dim * dtype_size;
}

static inline uint64_t ds3_kv_block_total_size(uint32_t block_size,
                                               uint32_t n_layer,
                                               uint32_t n_kv_head,
                                               uint32_t head_dim,
                                               uint32_t dtype_size)
{
    uint64_t kv = (uint64_t)block_size * ds3_kv_per_token_bytes(n_layer, n_kv_head, head_dim, dtype_size);
    return DS3_KVC_BLOCK_HEADER_SIZE + kv;
}

#ifdef __cplusplus
}
#endif

#endif /* DS3_KV_CACHE_H */
