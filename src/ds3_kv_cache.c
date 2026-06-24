/*
 * ds3 KV Cache provider implementations.
 *
 * Three providers are provided:
 *   - local:    in-memory per-session KV cache (default, backwards compatible)
 *   - fallback: no caching
 *   - service:  connects to ds3-kv-cache-svc over UDS + protobuf
 */

#include "ds3_kv_cache.h"
#include "kvcache/kvcache.pb.h"
#include "kvcache/pb_encode.h"
#include "kvcache/pb_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>

#define DS3_KVC_LOCAL_SESSIONS 16
#define DS3_KVC_LOCAL_TOKENS   8192
#define ds3_log_warn(...) fprintf(stderr, __VA_ARGS__)
#define DS3_KVC_LOCAL_KV_BYTES (DS3_KVC_LOCAL_TOKENS * 98304)
#define DS3_KVC_SOCKET_PATH    "/tmp/ds3-kv-cache.sock"

/* ========================================================================
 * Utility: xxhash64 (simple implementation for token hash verification)
 * ======================================================================== */

static const uint64_t XXH_PRIME64_1 = 11400714785074694791ULL;
static const uint64_t XXH_PRIME64_2 = 14029467366897019727ULL;
static const uint64_t XXH_PRIME64_3 = 1609587929392839161ULL;
static const uint64_t XXH_PRIME64_4 = 9650029242287828579ULL;
static const uint64_t XXH_PRIME64_5 = 2870177450012600261ULL;

static uint64_t xxh64_rotl(uint64_t x, int r)
{
    return (x << r) | (x >> (64 - r));
}

static uint64_t xxh64_round(uint64_t acc, uint64_t input)
{
    acc += input * XXH_PRIME64_2;
    acc  = xxh64_rotl(acc, 31);
    acc *= XXH_PRIME64_1;
    return acc;
}

static uint64_t xxh64_merge_round(uint64_t acc, uint64_t val)
{
    val  = xxh64_round(0, val);
    acc ^= val;
    acc  = acc * XXH_PRIME64_1 + XXH_PRIME64_4;
    return acc;
}

static uint64_t xxh64(const void *data, size_t len, uint64_t seed)
{
    const uint8_t *p = (const uint8_t *)data;
    const uint8_t *const bEnd = p + len;
    uint64_t h64;

    if (len >= 32) {
        const uint8_t *const limit = bEnd - 32;
        uint64_t v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        uint64_t v2 = seed + XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - XXH_PRIME64_1;

        do {
            v1 = xxh64_round(v1, *(const uint64_t *)(p + 0));
            v2 = xxh64_round(v2, *(const uint64_t *)(p + 8));
            v3 = xxh64_round(v3, *(const uint64_t *)(p + 16));
            v4 = xxh64_round(v4, *(const uint64_t *)(p + 24));
            p += 32;
        } while (p <= limit);

        h64 = xxh64_rotl(v1, 1) + xxh64_rotl(v2, 7) +
              xxh64_rotl(v3, 12) + xxh64_rotl(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + XXH_PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= bEnd) {
        uint64_t k1 = xxh64_round(0, *(const uint64_t *)p);
        h64 ^= k1;
        h64  = xxh64_rotl(h64, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
        p += 8;
    }

    if (p + 4 <= bEnd) {
        h64 ^= (uint64_t)(*(const uint32_t *)p) * XXH_PRIME64_1;
        h64  = xxh64_rotl(h64, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
        p += 4;
    }

    while (p < bEnd) {
        h64 ^= (uint64_t)(*p) * XXH_PRIME64_5;
        h64  = xxh64_rotl(h64, 11) * XXH_PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

/* ========================================================================
 * Fallback provider: no caching.
 * ======================================================================== */

typedef struct {
    int dummy;
} ds3_kv_fallback_state_t;

static ds3_kv_cache_provider_t *fallback_open(const char *config)
{
    (void)config;
    ds3_kv_cache_provider_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->vtable = &ds3_kv_fallback_provider;
    p->priv = calloc(1, sizeof(ds3_kv_fallback_state_t));
    if (!p->priv) { free(p); return NULL; }
    return p;
}

static void fallback_close(ds3_kv_cache_provider_t *p)
{
    if (!p) return;
    free(p->priv);
    free(p);
}

static int fallback_lookup(ds3_kv_cache_provider_t *p,
                           const char *session_id,
                           const int *tokens, int n_tokens,
                           int *out_cached_len)
{
    (void)p; (void)session_id; (void)tokens; (void)n_tokens;
    *out_cached_len = 0;
    return 0;
}

static int fallback_read(ds3_kv_cache_provider_t *p,
                         const char *session_id,
                         int start_pos, int n_tokens,
                         void *kv_buffer)
{
    (void)p; (void)session_id; (void)start_pos; (void)n_tokens; (void)kv_buffer;
    return 0;
}

static int fallback_write(ds3_kv_cache_provider_t *p,
                          const char *session_id,
                          const int *tokens, int n_tokens,
                          const void *kv_buffer, size_t kv_buffer_bytes)
{
    (void)p; (void)session_id; (void)tokens; (void)n_tokens;
    (void)kv_buffer; (void)kv_buffer_bytes;
    return 0;
}

static int fallback_create_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    (void)p; (void)session_id;
    return 0;
}

static int fallback_close_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    (void)p; (void)session_id;
    return 0;
}

static int fallback_reset_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    (void)p; (void)session_id;
    return 0;
}

const struct ds3_kv_cache_provider_vtable ds3_kv_fallback_provider = {
    .name = "fallback",
    .open = fallback_open,
    .close = fallback_close,
    .lookup = fallback_lookup,
    .read = fallback_read,
    .write = fallback_write,
    .create_session = fallback_create_session,
    .close_session = fallback_close_session,
    .reset_session = fallback_reset_session,
};

/* ========================================================================
 * Local provider: in-memory per-session cache.
 * ======================================================================== */

typedef struct {
    char session_id[64];
    int valid;
    int *tokens;
    int n_tokens;
    int capacity_tokens;
    uint8_t *kv_data;
    size_t kv_capacity;
} ds3_kv_local_session_t;

typedef struct {
    ds3_kv_local_session_t sessions[DS3_KVC_LOCAL_SESSIONS];
    int per_token_kv_bytes;
} ds3_kv_local_state_t;

static ds3_kv_local_session_t *local_find_session(ds3_kv_local_state_t *st, const char *session_id)
{
    for (int i = 0; i < DS3_KVC_LOCAL_SESSIONS; i++) {
        if (st->sessions[i].valid && strcmp(st->sessions[i].session_id, session_id) == 0) {
            return &st->sessions[i];
        }
    }
    return NULL;
}

static ds3_kv_local_session_t *local_alloc_session(ds3_kv_local_state_t *st)
{
    for (int i = 0; i < DS3_KVC_LOCAL_SESSIONS; i++) {
        if (!st->sessions[i].valid) return &st->sessions[i];
    }
    return NULL;
}

static ds3_kv_cache_provider_t *local_open(const char *config)
{
    (void)config;
    ds3_kv_cache_provider_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    ds3_kv_local_state_t *st = calloc(1, sizeof(*st));
    if (!st) { free(p); return NULL; }

    st->per_token_kv_bytes = 98304; /* Qwen3-30B-A3B default */

    /* Session slots start unallocated; buffers are lazy-allocated in
     * local_create_session to avoid pre-allocating ~12.9 GB at startup. */

    p->vtable = &ds3_kv_local_provider;
    p->priv = st;
    return p;
}

static void local_release_session(ds3_kv_local_session_t *s)
{
    free(s->tokens);
    free(s->kv_data);
    s->tokens = NULL;
    s->kv_data = NULL;
    s->n_tokens = 0;
    s->capacity_tokens = 0;
    s->kv_capacity = 0;
}

static void local_close(ds3_kv_cache_provider_t *p)
{
    if (!p) return;
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    if (st) {
        for (int i = 0; i < DS3_KVC_LOCAL_SESSIONS; i++) {
            local_release_session(&st->sessions[i]);
        }
        free(st);
    }
    free(p);
}

static int local_create_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    ds3_kv_local_session_t *s = local_find_session(st, session_id);
    if (s) {
        s->n_tokens = 0;
        return 0;
    }
    s = local_alloc_session(st);
    if (!s) return -1;
    strncpy(s->session_id, session_id, sizeof(s->session_id) - 1);
    s->session_id[sizeof(s->session_id) - 1] = '\0';
    /* Lazy-allocate buffers (avoids ~12.9 GB pre-allocation at provider open). */
    s->tokens = calloc(DS3_KVC_LOCAL_TOKENS, sizeof(int));
    s->kv_data = calloc(DS3_KVC_LOCAL_KV_BYTES, 1);
    s->capacity_tokens = DS3_KVC_LOCAL_TOKENS;
    s->kv_capacity = DS3_KVC_LOCAL_KV_BYTES;
    if (!s->tokens || !s->kv_data) {
        local_release_session(s);
        return -1;
    }
    s->valid = 1;
    s->n_tokens = 0;
    return 0;
}

static int local_close_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    ds3_kv_local_session_t *s = local_find_session(st, session_id);
    if (s) {
        local_release_session(s);
        s->valid = 0;
    }
    return 0;
}

static int local_reset_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    ds3_kv_local_session_t *s = local_find_session(st, session_id);
    if (s) s->n_tokens = 0;
    return 0;
}

static int local_lookup(ds3_kv_cache_provider_t *p,
                        const char *session_id,
                        const int *tokens, int n_tokens,
                        int *out_cached_len)
{
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    ds3_kv_local_session_t *s = local_find_session(st, session_id);
    *out_cached_len = 0;
    if (!s || s->n_tokens == 0) return 0;

    int max = (n_tokens < s->n_tokens) ? n_tokens : s->n_tokens;
    int i = 0;
    while (i < max && tokens[i] == s->tokens[i]) i++;
    *out_cached_len = i;
    return 0;
}

static int local_read(ds3_kv_cache_provider_t *p,
                      const char *session_id,
                      int start_pos, int n_tokens,
                      void *kv_buffer)
{
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    ds3_kv_local_session_t *s = local_find_session(st, session_id);
    if (!s) return -1;
    if (start_pos + n_tokens > s->n_tokens) return -1;

    int bytes = n_tokens * st->per_token_kv_bytes;
    memcpy(kv_buffer, s->kv_data + (size_t)start_pos * st->per_token_kv_bytes, bytes);
    return 0;
}

static int local_write(ds3_kv_cache_provider_t *p,
                       const char *session_id,
                       const int *tokens, int n_tokens,
                       const void *kv_buffer, size_t kv_buffer_bytes)
{
    ds3_kv_local_state_t *st = (ds3_kv_local_state_t *)p->priv;
    ds3_kv_local_session_t *s = local_find_session(st, session_id);
    if (!s) return -1;
    if (n_tokens > s->capacity_tokens) return -1;
    if (kv_buffer_bytes > s->kv_capacity) return -1;

    memcpy(s->tokens, tokens, (size_t)n_tokens * sizeof(int));
    memcpy(s->kv_data, kv_buffer, kv_buffer_bytes);
    s->n_tokens = n_tokens;
    return 0;
}

const struct ds3_kv_cache_provider_vtable ds3_kv_local_provider = {
    .name = "local",
    .open = local_open,
    .close = local_close,
    .lookup = local_lookup,
    .read = local_read,
    .write = local_write,
    .create_session = local_create_session,
    .close_session = local_close_session,
    .reset_session = local_reset_session,
};

/* ========================================================================
 * Service provider: connects to ds3-kv-cache-svc over UDS + shared mmap.
 * ======================================================================== */

#define DS3_KVC_SERVICE_MAX_BLOCKS 4096
#define DS3_KVC_SERVICE_SESSIONS   64

typedef struct {
    ds3_kv_block_handle_t handle;
    int start_token;   /* inclusive token position this block covers */
    int n_tokens;      /* actual token count stored in this block */
} service_block_t;

typedef struct {
    char session_id[64];
    int valid;
    int cached_len;
    service_block_t blocks[DS3_KVC_SERVICE_MAX_BLOCKS];
    int n_blocks;
} service_session_cache_t;

typedef struct {
    int fd;
    char socket_path[256];
    int connected;

    /* mmap */
    char mmap_path[256];
    int mmap_fd;
    uint8_t *mmap_base;
    size_t mmap_size;
    ds3_kv_region_header_t *region_header;

    uint32_t block_size;       /* tokens per block */
    uint64_t block_total_size; /* header + kv data */
    uint64_t block_data_size;  /* kv data bytes per block */
    int per_token_kv_bytes;

    /* Cached lookup results per session. */
    service_session_cache_t sessions[DS3_KVC_SERVICE_SESSIONS];

    /* Heartbeat lease tracking. */
    char daemon_id[32];
    time_t last_heartbeat;

    /* Background heartbeat thread.
     * - heartbeat_thread: detached at creation, joined in service_close.
     * - shutdown: set to 1 by service_close to wake the thread immediately.
     * - wake_cond / wake_mutex: signalled every HEARTBEAT_INTERVAL_SECS, or
     *   on shutdown, so the thread does not have to poll sleep(10) and can
     *   exit promptly. */
    pthread_t heartbeat_thread;
    int shutdown;
    pthread_mutex_t wake_mutex;
    pthread_cond_t wake_cond;
} ds3_kv_service_state_t;

#define HEARTBEAT_INTERVAL_SECS 10

static int service_heartbeat(ds3_kv_service_state_t *svc);

/* Forward declaration: service_call_inner is used by service_call (above) and
 * service_heartbeat (below).  The full definition appears later in the file. */
static int service_call_inner(ds3_kv_service_state_t *svc,
                              const char *method,
                              const pb_msgdesc_t *req_fields, const void *req_msg,
                              const pb_msgdesc_t *resp_fields, void *resp_msg);

/* Called by the main thread and the heartbeat thread.  Serialises all
 * RPC activity so that two concurrent threads (e.g. an in-flight generate
 * RPC and a periodic heartbeat) do not interleave on the same socket. */
static int service_call(ds3_kv_service_state_t *svc,
                        const char *method,
                        const pb_msgdesc_t *req_fields, const void *req_msg,
                        const pb_msgdesc_t *resp_fields, void *resp_msg)
{
    pthread_mutex_lock(&svc->wake_mutex);
    int rc = service_call_inner(svc, method, req_fields, req_msg, resp_fields, resp_msg);
    pthread_mutex_unlock(&svc->wake_mutex);
    return rc;
}

static int service_connect(ds3_kv_service_state_t *svc)
{
    if (svc->connected) {
        return 0;
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, svc->socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    svc->fd = fd;
    svc->connected = 1;
    svc->last_heartbeat = time(NULL);
    return 0;
}

static void service_disconnect(ds3_kv_service_state_t *svc)
{
    if (svc->fd >= 0) {
        close(svc->fd);
        svc->fd = -1;
    }
    svc->connected = 0;
}

/* Send a full protobuf message with length prefix. */
static int service_send(int fd, const uint8_t *data, size_t len)
{
    uint8_t header[4];
    header[0] = (len >> 24) & 0xFF;
    header[1] = (len >> 16) & 0xFF;
    header[2] = (len >> 8) & 0xFF;
    header[3] = len & 0xFF;

    size_t total = 4 + len;
    uint8_t *buf = malloc(total);
    if (!buf) return -1;
    memcpy(buf, header, 4);
    memcpy(buf + 4, data, len);

    size_t sent = 0;
    while (sent < total) {
        ssize_t n = write(fd, buf + sent, total - sent);
        if (n <= 0) { free(buf); return -1; }
        sent += n;
    }
    free(buf);
    return 0;
}

/* Receive a full protobuf message. Caller frees *out_data. */
static int service_recv(int fd, uint8_t **out_data, size_t *out_len)
{
    uint8_t header[4];
    size_t got = 0;
    while (got < 4) {
        ssize_t n = read(fd, header + got, 4 - got);
        if (n <= 0) return -1;
        got += n;
    }
    size_t len = ((size_t)header[0] << 24) |
                 ((size_t)header[1] << 16) |
                 ((size_t)header[2] << 8) |
                 (size_t)header[3];

    uint8_t *data = malloc(len);
    if (!data) return -1;

    got = 0;
    while (got < len) {
        ssize_t n = read(fd, data + got, len - got);
        if (n <= 0) { free(data); return -1; }
        got += n;
    }

    *out_data = data;
    *out_len = len;
    return 0;
}

/* ------------------------------------------------------------------------
 * nanopb callbacks
 * ------------------------------------------------------------------------ */

typedef struct {
    const int *tokens;
    int n_tokens;
    int pos;
} encode_int32_ctx_t;

typedef struct {
    const uint64_t *values;
    int n_values;
    int pos;
} encode_uint64_ctx_t;

typedef struct {
    char *buf;
    size_t buf_size;
} decode_string_ctx_t;

typedef struct {
    service_block_t *blocks;
    int max_blocks;
    int n_blocks;
} decode_blocks_ctx_t;

static bool encode_string(pb_ostream_t *stream, const pb_field_t *field, void *const *arg)
{
    const char *str = (const char *)*arg;
    if (!pb_encode_tag_for_field(stream, field)) return false;
    return pb_encode_string(stream, (const pb_byte_t *)str, strlen(str));
}

static bool encode_bytes(pb_ostream_t *stream, const pb_field_t *field, void *const *arg)
{
    const pb_bytes_array_t *bytes = (const pb_bytes_array_t *)*arg;
    if (!pb_encode_tag_for_field(stream, field)) return false;
    return pb_encode_string(stream, bytes->bytes, bytes->size);
}

static bool encode_repeated_int32(pb_ostream_t *stream, const pb_field_t *field, void *const *arg)
{
    encode_int32_ctx_t *ctx = (encode_int32_ctx_t *)*arg;
    for (int i = 0; i < ctx->n_tokens; i++) {
        if (!pb_encode_tag_for_field(stream, field)) return false;
        if (!pb_encode_varint(stream, (uint64_t)(int64_t)ctx->tokens[i])) return false;
    }
    return true;
}

static bool encode_repeated_uint64(pb_ostream_t *stream, const pb_field_t *field, void *const *arg)
{
    encode_uint64_ctx_t *ctx = (encode_uint64_ctx_t *)*arg;
    for (int i = 0; i < ctx->n_values; i++) {
        if (!pb_encode_tag_for_field(stream, field)) return false;
        if (!pb_encode_varint(stream, ctx->values[i])) return false;
    }
    return true;
}

/* Callback context for encoding repeated strings (heartbeat session_ids). */
typedef struct {
    char (*strings)[64];
    int n_strings;
} encode_strings_ctx_t;

static bool encode_repeated_string(pb_ostream_t *stream, const pb_field_t *field, void *const *arg)
{
    encode_strings_ctx_t *ctx = (encode_strings_ctx_t *)*arg;
    for (int i = 0; i < ctx->n_strings; i++) {
        if (!pb_encode_tag_for_field(stream, field)) return false;
        if (!pb_encode_string(stream, (const pb_byte_t *)ctx->strings[i], strlen(ctx->strings[i]))) return false;
    }
    return true;
}

static bool decode_string_to_buf(pb_istream_t *stream, const pb_field_t *field, void **arg)
{
    (void)field;
    decode_string_ctx_t *ctx = (decode_string_ctx_t *)*arg;
    size_t len = stream->bytes_left;
    if (len >= ctx->buf_size) len = ctx->buf_size - 1;
    if (!pb_read(stream, (pb_byte_t *)ctx->buf, len)) return false;
    ctx->buf[len] = '\0';
    if (stream->bytes_left > 0) {
        if (!pb_read(stream, NULL, stream->bytes_left)) return false;
    }
    return true;
}

static bool decode_repeated_block_handle(pb_istream_t *stream, const pb_field_t *field, void **arg)
{
    (void)field;
    decode_blocks_ctx_t *ctx = (decode_blocks_ctx_t *)*arg;
    if (ctx->n_blocks >= ctx->max_blocks) return false;
    ds3_kvcache_BlockHandle bh = ds3_kvcache_BlockHandle_init_zero;
    if (!pb_decode(stream, ds3_kvcache_BlockHandle_fields, &bh)) return false;
    ctx->blocks[ctx->n_blocks].handle.block_id = bh.block_id;
    ctx->blocks[ctx->n_blocks].handle.mmap_offset = bh.mmap_offset;
    ctx->blocks[ctx->n_blocks].start_token = 0;
    ctx->blocks[ctx->n_blocks].n_tokens = 0;
    ctx->n_blocks++;
    return true;
}

/* Wrapper for RpcRequest envelope with dynamic payload sizing. */
typedef struct {
    pb_size_t size;
    pb_byte_t bytes[];
} pb_bytes_dyn_t;

/* Caller must hold svc->wake_mutex. */
static int service_call_inner(ds3_kv_service_state_t *svc,
                              const char *method,
                              const pb_msgdesc_t *req_fields, const void *req_msg,
                              const pb_msgdesc_t *resp_fields, void *resp_msg)
{
    if (service_connect(svc) != 0) return -1;

    size_t payload_cap = 4096;
    uint8_t *payload = (uint8_t *)malloc(payload_cap);
    if (!payload) return -1;

    pb_ostream_t pstream;
    for (;;) {
        pstream = pb_ostream_from_buffer(payload, payload_cap);
        if (pb_encode(&pstream, req_fields, req_msg)) break;
        if (pstream.errmsg && strstr(pstream.errmsg, "stream full")) {
            payload_cap *= 2;
            uint8_t *tmp = (uint8_t *)realloc(payload, payload_cap);
            if (!tmp) { free(payload); return -1; }
            payload = tmp;
            continue;
        }
        free(payload);
        return -1;
    }
    size_t payload_len = pstream.bytes_written;

    ds3_kvcache_RpcRequest envelope = ds3_kvcache_RpcRequest_init_zero;
    envelope.method.funcs.encode = &encode_string;
    envelope.method.arg = (void *)method;
    envelope.payload.funcs.encode = &encode_bytes;

    pb_bytes_dyn_t *payload_bytes = (pb_bytes_dyn_t *)malloc(sizeof(pb_bytes_dyn_t) + payload_len);
    if (!payload_bytes) { free(payload); return -1; }
    payload_bytes->size = (pb_size_t)payload_len;
    memcpy(payload_bytes->bytes, payload, payload_len);
    envelope.payload.arg = payload_bytes;

    size_t enc_cap = 8192;
    uint8_t *enc = (uint8_t *)malloc(enc_cap);
    if (!enc) { free(payload); free(payload_bytes); return -1; }

    pb_ostream_t estream;
    for (;;) {
        estream = pb_ostream_from_buffer(enc, enc_cap);
        if (pb_encode(&estream, ds3_kvcache_RpcRequest_fields, &envelope)) break;
        if (estream.errmsg && strstr(estream.errmsg, "stream full")) {
            enc_cap *= 2;
            uint8_t *tmp = (uint8_t *)realloc(enc, enc_cap);
            if (!tmp) { free(payload); free(payload_bytes); free(enc); return -1; }
            enc = tmp;
            continue;
        }
        free(payload); free(payload_bytes); free(enc);
        return -1;
    }
    free(payload);
    free(payload_bytes);

    if (service_send(svc->fd, enc, estream.bytes_written) != 0) {
        service_disconnect(svc);
        free(enc);
        return -1;
    }
    free(enc);

    uint8_t *resp_data = NULL;
    size_t resp_len = 0;
    if (service_recv(svc->fd, &resp_data, &resp_len) != 0) {
        service_disconnect(svc);
        return -1;
    }

    pb_istream_t istream = pb_istream_from_buffer(resp_data, resp_len);
    bool ok = pb_decode(&istream, resp_fields, resp_msg);
    free(resp_data);
    return ok ? 0 : -1;
}

/* Send a Heartbeat RPC to the service, listing all active sessions.
 * Caller must hold svc->wake_mutex for the entire call so the session
 * scan and RPC are atomic with respect to concurrent close_session / reset_session. */
static int service_heartbeat(ds3_kv_service_state_t *svc)
{
    char session_ids[DS3_KVC_SERVICE_SESSIONS][64];
    int n_sessions = 0;
    for (int i = 0; i < DS3_KVC_SERVICE_SESSIONS; i++) {
        if (svc->sessions[i].valid && svc->sessions[i].session_id[0]) {
            strncpy(session_ids[n_sessions], svc->sessions[i].session_id, 63);
            session_ids[n_sessions][63] = '\0';
            n_sessions++;
        }
    }

    encode_strings_ctx_t ctx = { .strings = session_ids, .n_strings = n_sessions };

    ds3_kvcache_HeartbeatReq req = ds3_kvcache_HeartbeatReq_init_zero;
    req.daemon_id.funcs.encode = &encode_string;
    req.daemon_id.arg = (void *)svc->daemon_id;
    req.session_ids.funcs.encode = &encode_repeated_string;
    req.session_ids.arg = &ctx;

    ds3_kvcache_StatusResp resp = ds3_kvcache_StatusResp_init_zero;
    int rc = service_call_inner(svc, "Heartbeat",
                                ds3_kvcache_HeartbeatReq_fields, &req,
                                ds3_kvcache_StatusResp_fields, &resp);
    if (rc == 0 && resp.ok) {
        svc->last_heartbeat = time(NULL);
    }
    return rc;
}

/* Wait up to `secs` for a wake event or shutdown.  Returns 1 if shutdown
 * was signalled, 0 if timeout elapsed. */
static int wait_or_shutdown(ds3_kv_service_state_t *svc, int secs)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += secs;
    pthread_mutex_lock(&svc->wake_mutex);
    while (!svc->shutdown) {
        int rc = pthread_cond_timedwait(&svc->wake_cond, &svc->wake_mutex, &ts);
        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&svc->wake_mutex);
            return 0;
        }
        if (rc != 0 && rc != EINTR) {
            pthread_mutex_unlock(&svc->wake_mutex);
            return 0;
        }
        if (svc->shutdown) {
            pthread_mutex_unlock(&svc->wake_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&svc->wake_mutex);
    return 1;
}

static void *service_heartbeat_thread(void *arg)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)arg;
    while (!svc->shutdown) {
        if (wait_or_shutdown(svc, HEARTBEAT_INTERVAL_SECS)) {
            break;  /* shutdown signalled */
        }
        /* Send heartbeat holding the mutex so session scan and RPC are atomic. */
        pthread_mutex_lock(&svc->wake_mutex);
        service_heartbeat(svc);
        pthread_mutex_unlock(&svc->wake_mutex);
    }
    return NULL;
}

/* ------------------------------------------------------------------------
 * mmap management
 * ------------------------------------------------------------------------ */

static int service_ensure_mmap(ds3_kv_service_state_t *svc)
{
    if (svc->mmap_base) return 0;

    ds3_kvcache_Empty req = ds3_kvcache_Empty_init_zero;
    ds3_kvcache_ConfigResp resp = ds3_kvcache_ConfigResp_init_zero;
    char mmap_path[256] = {0};
    char err_buf[256] = {0};
    decode_string_ctx_t path_ctx = { mmap_path, sizeof(mmap_path) };
    decode_string_ctx_t err_ctx = { err_buf, sizeof(err_buf) };
    resp.mmap_path.funcs.decode = &decode_string_to_buf;
    resp.mmap_path.arg = &path_ctx;
    resp.error.funcs.decode = &decode_string_to_buf;
    resp.error.arg = &err_ctx;

    if (service_call(svc, "GetConfig",
                     ds3_kvcache_Empty_fields, &req,
                     ds3_kvcache_ConfigResp_fields, &resp) != 0) {
        return -1;
    }
    if (!resp.ok) {
        ds3_log_warn("kv-cache GetConfig failed: %s\n", err_buf[0] ? err_buf : "unknown");
        return -1;
    }

    strncpy(svc->mmap_path, mmap_path, sizeof(svc->mmap_path) - 1);
    svc->block_size = resp.block_size ? resp.block_size : 16;
    svc->per_token_kv_bytes = (int)resp.per_token_kv_bytes;
    if (svc->per_token_kv_bytes <= 0) svc->per_token_kv_bytes = 98304;
    svc->block_data_size = (uint64_t)svc->block_size * (uint64_t)svc->per_token_kv_bytes;
    svc->block_total_size = svc->block_data_size + DS3_KVC_BLOCK_HEADER_SIZE;
    svc->mmap_size = (size_t)resp.mmap_size;
    if (svc->mmap_size == 0) svc->mmap_size = (size_t)svc->block_total_size * 1024 + DS3_KVC_REGION_HEADER_SIZE;

    int fd = open(svc->mmap_path, O_RDWR);
    if (fd < 0) return -1;
    svc->mmap_fd = fd;

    uint8_t *base = (uint8_t *)mmap(NULL, svc->mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        svc->mmap_fd = -1;
        return -1;
    }
    svc->mmap_base = base;
    svc->region_header = (ds3_kv_region_header_t *)base;

    if (svc->region_header->magic != DS3_KVC_REGION_MAGIC ||
        svc->region_header->version != DS3_KVC_VERSION) {
        munmap(base, svc->mmap_size);
        close(fd);
        svc->mmap_base = NULL;
        svc->mmap_fd = -1;
        return -1;
    }

    svc->block_size = svc->region_header->block_size ? svc->region_header->block_size : svc->block_size;
    svc->block_data_size = svc->region_header->block_data_size ? svc->region_header->block_data_size : svc->block_data_size;
    svc->block_total_size = svc->block_data_size + DS3_KVC_BLOCK_HEADER_SIZE;
    return 0;
}

/* Caller must hold svc->wake_mutex. */
static service_session_cache_t *service_find_session_cache_unlocked(ds3_kv_service_state_t *svc, const char *session_id)
{
    for (int i = 0; i < DS3_KVC_SERVICE_SESSIONS; i++) {
        if (svc->sessions[i].valid && strcmp(svc->sessions[i].session_id, session_id) == 0) {
            return &svc->sessions[i];
        }
    }
    return NULL;
}

/* Protects svc->sessions[] against concurrent access from the heartbeat
 * thread and the main engine thread.  Uses wake_mutex because it is also
 * used to serialise all RPC activity on the daemon-side socket. */
static service_session_cache_t *service_find_session_cache(ds3_kv_service_state_t *svc, const char *session_id)
{
    pthread_mutex_lock(&svc->wake_mutex);
    for (int i = 0; i < DS3_KVC_SERVICE_SESSIONS; i++) {
        if (svc->sessions[i].valid && strcmp(svc->sessions[i].session_id, session_id) == 0) {
            pthread_mutex_unlock(&svc->wake_mutex);
            return &svc->sessions[i];
        }
    }
    pthread_mutex_unlock(&svc->wake_mutex);
    return NULL;
}

static service_session_cache_t *service_alloc_session_cache(ds3_kv_service_state_t *svc, const char *session_id)
{
    pthread_mutex_lock(&svc->wake_mutex);
    service_session_cache_t *s = service_find_session_cache_unlocked(svc, session_id);
    if (s) {
        pthread_mutex_unlock(&svc->wake_mutex);
        return s;
    }
    for (int i = 0; i < DS3_KVC_SERVICE_SESSIONS; i++) {
        if (!svc->sessions[i].valid) {
            svc->sessions[i].valid = 1;
            strncpy(svc->sessions[i].session_id, session_id, sizeof(svc->sessions[i].session_id) - 1);
            svc->sessions[i].session_id[sizeof(svc->sessions[i].session_id) - 1] = '\0';
            svc->sessions[i].cached_len = 0;
            svc->sessions[i].n_blocks = 0;
            pthread_mutex_unlock(&svc->wake_mutex);
            return &svc->sessions[i];
        }
    }
    pthread_mutex_unlock(&svc->wake_mutex);
    return NULL;
}

static uint64_t service_token_hash(const int *tokens, int n)
{
    if (n <= 0) return xxh64(NULL, 0, 0);
    /* Hash the full little-endian bytes of each i32 token, matching the Rust
     * service's token_hash() helper. */
    uint8_t bytes[1024];
    uint8_t *p = bytes;
    size_t need = (size_t)n * sizeof(int);
    if (need > sizeof(bytes)) p = (uint8_t *)malloc(need);
    if (!p) return 0;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)tokens[i];
        p[i * 4 + 0] = (uint8_t)(v & 0xFF);
        p[i * 4 + 1] = (uint8_t)((v >> 8) & 0xFF);
        p[i * 4 + 2] = (uint8_t)((v >> 16) & 0xFF);
        p[i * 4 + 3] = (uint8_t)((v >> 24) & 0xFF);
    }
    uint64_t h = xxh64(p, need, 0);
    if (need > sizeof(bytes)) free(p);
    return h;
}

static ds3_kv_cache_provider_t *service_open(const char *config)
{
    ds3_kv_cache_provider_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    ds3_kv_service_state_t *svc = calloc(1, sizeof(*svc));
    if (!svc) { free(p); return NULL; }

    svc->fd = -1;
    svc->mmap_fd = -1;
    svc->last_heartbeat = 0;
    svc->shutdown = 0;
    snprintf(svc->daemon_id, sizeof(svc->daemon_id), "%d", (int)getpid());
    if (config && config[0]) {
        strncpy(svc->socket_path, config, sizeof(svc->socket_path) - 1);
    } else {
        strncpy(svc->socket_path, DS3_KVC_SOCKET_PATH, sizeof(svc->socket_path) - 1);
    }

    /* Non-recursive mutex: wake_cond is signalled while holding it to wake
     * the heartbeat thread immediately on shutdown. */
    if (pthread_mutex_init(&svc->wake_mutex, NULL) != 0) {
        free(svc);
        free(p);
        return NULL;
    }
    if (pthread_cond_init(&svc->wake_cond, NULL) != 0) {
        pthread_mutex_destroy(&svc->wake_mutex);
        free(svc);
        free(p);
        return NULL;
    }

    if (pthread_create(&svc->heartbeat_thread, NULL, service_heartbeat_thread, svc) != 0) {
        pthread_cond_destroy(&svc->wake_cond);
        pthread_mutex_destroy(&svc->wake_mutex);
        free(svc);
        free(p);
        return NULL;
    }

    p->vtable = &ds3_kv_service_provider;
    p->priv = svc;
    return p;
}

static void service_close(ds3_kv_cache_provider_t *p)
{
    if (!p) return;
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    if (svc) {
        /* Signal shutdown and wake the heartbeat thread immediately so
         * pthread_join does not have to wait for the next tick. */
        pthread_mutex_lock(&svc->wake_mutex);
        svc->shutdown = 1;
        pthread_cond_broadcast(&svc->wake_cond);
        pthread_mutex_unlock(&svc->wake_mutex);
        pthread_join(svc->heartbeat_thread, NULL);
        pthread_cond_destroy(&svc->wake_cond);
        pthread_mutex_destroy(&svc->wake_mutex);
        service_disconnect(svc);
        if (svc->mmap_base) munmap(svc->mmap_base, svc->mmap_size);
        if (svc->mmap_fd >= 0) close(svc->mmap_fd);
        free(svc);
    }
    free(p);
}

static int service_create_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    ds3_kvcache_CreateSessionReq req = ds3_kvcache_CreateSessionReq_init_zero;
    req.session_id.funcs.encode = &encode_string;
    req.session_id.arg = (void *)session_id;
    ds3_kvcache_StatusResp resp = ds3_kvcache_StatusResp_init_zero;
    int rc = service_call(svc, "CreateSession",
                          ds3_kvcache_CreateSessionReq_fields, &req,
                          ds3_kvcache_StatusResp_fields, &resp);
    if (rc != 0) return rc;
    service_session_cache_t *sc = service_find_session_cache(svc, session_id);
    if (sc) {
        sc->cached_len = 0;
        sc->n_blocks = 0;
    }
    return resp.ok ? 0 : -1;
}

static int service_close_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    ds3_kvcache_CloseSessionReq req = ds3_kvcache_CloseSessionReq_init_zero;
    req.session_id.funcs.encode = &encode_string;
    req.session_id.arg = (void *)session_id;
    ds3_kvcache_StatusResp resp = ds3_kvcache_StatusResp_init_zero;
    int rc = service_call(svc, "CloseSession",
                          ds3_kvcache_CloseSessionReq_fields, &req,
                          ds3_kvcache_StatusResp_fields, &resp);
    service_session_cache_t *sc = service_find_session_cache(svc, session_id);
    if (sc) sc->valid = 0;
    return (rc == 0 && resp.ok) ? 0 : -1;
}

static int service_reset_session(ds3_kv_cache_provider_t *p, const char *session_id)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    ds3_kvcache_ResetSessionReq req = ds3_kvcache_ResetSessionReq_init_zero;
    req.session_id.funcs.encode = &encode_string;
    req.session_id.arg = (void *)session_id;
    ds3_kvcache_StatusResp resp = ds3_kvcache_StatusResp_init_zero;
    int rc = service_call(svc, "ResetSession",
                          ds3_kvcache_ResetSessionReq_fields, &req,
                          ds3_kvcache_StatusResp_fields, &resp);
    if (rc != 0) return rc;
    service_session_cache_t *sc = service_find_session_cache(svc, session_id);
    if (sc) {
        sc->cached_len = 0;
        sc->n_blocks = 0;
    }
    return resp.ok ? 0 : -1;
}

static int service_lookup(ds3_kv_cache_provider_t *p,
                          const char *session_id,
                          const int *tokens, int n_tokens,
                          int *out_cached_len)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    *out_cached_len = 0;
    if (service_ensure_mmap(svc) != 0) return 0;

    ds3_kvcache_LookupReq req = ds3_kvcache_LookupReq_init_zero;
    req.session_id.funcs.encode = &encode_string;
    req.session_id.arg = (void *)session_id;

    encode_int32_ctx_t tokens_ctx = { tokens, n_tokens, 0 };
    req.tokens.funcs.encode = &encode_repeated_int32;
    req.tokens.arg = &tokens_ctx;

    service_session_cache_t *sc = service_alloc_session_cache(svc, session_id);
    if (!sc) return 0;
    decode_blocks_ctx_t blocks_ctx = { sc->blocks, DS3_KVC_SERVICE_MAX_BLOCKS, 0 };

    ds3_kvcache_LookupResp resp = ds3_kvcache_LookupResp_init_zero;
    resp.blocks.funcs.decode = &decode_repeated_block_handle;
    resp.blocks.arg = &blocks_ctx;
    char err_buf[256] = {0};
    decode_string_ctx_t err_ctx = { err_buf, sizeof(err_buf) };
    resp.error.funcs.decode = &decode_string_to_buf;
    resp.error.arg = &err_ctx;

    if (service_call(svc, "Lookup",
                     ds3_kvcache_LookupReq_fields, &req,
                     ds3_kvcache_LookupResp_fields, &resp) != 0) {
        sc->cached_len = 0;
        sc->n_blocks = 0;
        return 0;
    }
    if (!resp.ok) {
        ds3_log_warn("kv-cache Lookup failed: %s\n", err_buf[0] ? err_buf : "unknown");
        sc->cached_len = 0;
        sc->n_blocks = 0;
        return 0;
    }

    /* Defense layer 2 (§10.4): verify the returned prefix hash against the
     * local token sequence.  Mismatch → fall back to full prefill. */
    if (resp.cached_prefix_len > 0) {
        uint64_t local_hash = xxh64(tokens,
                                    (size_t)resp.cached_prefix_len * sizeof(int),
                                    0);
        if (local_hash != resp.prefix_token_hash) {
            ds3_log_warn("kv-cache prefix token hash mismatch: local=%llu remote=%llu\n",
                         (unsigned long long)local_hash,
                         (unsigned long long)resp.prefix_token_hash);
            sc->cached_len = 0;
            sc->n_blocks = 0;
            *out_cached_len = 0;
            return 0;
        }
    }

    sc->cached_len = resp.cached_prefix_len;
    sc->n_blocks = blocks_ctx.n_blocks;

    /* Read actual token counts from block headers so partial blocks are
     * handled correctly when a prefix spans multiple store operations. */
    int pos = 0;
    for (int i = 0; i < sc->n_blocks; i++) {
        sc->blocks[i].start_token = pos;
        size_t header_off = sc->blocks[i].handle.mmap_offset - DS3_KVC_BLOCK_HEADER_SIZE;
        ds3_kv_block_header_t *bh = (ds3_kv_block_header_t *)(svc->mmap_base + header_off);
        if (bh->magic == DS3_KVC_BLOCK_MAGIC && bh->state == DS3_KVC_BLOCK_VALID) {
            sc->blocks[i].n_tokens = bh->n_tokens;
        } else {
            /* Fallback: assume full block except possibly the last one. */
            int remaining = sc->cached_len - pos;
            int chunk = (int)svc->block_size < remaining ? (int)svc->block_size : remaining;
            sc->blocks[i].n_tokens = chunk;
        }
        pos += sc->blocks[i].n_tokens;
    }

    *out_cached_len = sc->cached_len;
    return 0;
}

static int service_read(ds3_kv_cache_provider_t *p,
                        const char *session_id,
                        int start_pos, int n_tokens,
                        void *kv_buffer)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    if (!svc->mmap_base || !kv_buffer) return -1;
    service_session_cache_t *sc = service_find_session_cache(svc, session_id);
    if (!sc || sc->cached_len < start_pos + n_tokens) return -1;

    uint8_t *dst = (uint8_t *)kv_buffer;
    /* Host buffer is layer-interleaved:
     *   [L0_K(tok 0..P-1)][L0_V(tok 0..P-1)]...[L47_V(tok 0..P-1)]
     * mmap blocks are token-interleaved (matching Rust BlockStore layout):
     *   [tok0(all layers K+V)][tok1(all layers K+V)]...
     * so we transpose block -> host here. */
    int ptl_bytes = (int)svc->region_header->n_kv_head
                  * (int)svc->region_header->head_dim * (int)DS3_KVC_DEFAULT_DTYPE_SIZE;
    int n_layer = (int)svc->region_header->n_layer;
    int copied = 0;
    for (int i = 0; i < sc->n_blocks && copied < n_tokens; i++) {
        service_block_t *b = &sc->blocks[i];
        int block_end = b->start_token + b->n_tokens;
        int read_start = start_pos > b->start_token ? start_pos : b->start_token;
        int read_end = (start_pos + n_tokens) < block_end ? (start_pos + n_tokens) : block_end;
        if (read_start >= read_end) continue;

        int tok_offset = read_start - b->start_token;
        int n = read_end - read_start;

        for (int t = 0; t < n; t++) {
            int global_token = read_start + t;
            int out_token = global_token - start_pos;
            size_t src_token_base = b->handle.mmap_offset
                                  + (size_t)(tok_offset + t) * svc->per_token_kv_bytes;
            for (int l = 0; l < n_layer; l++) {
                /* K */
                size_t src_k = src_token_base + (size_t)(2*l) * ptl_bytes;
                size_t dst_k = (size_t)(2*l) * n_tokens * ptl_bytes
                             + (size_t)out_token * ptl_bytes;
                memcpy(dst + dst_k, svc->mmap_base + src_k, (size_t)ptl_bytes);
                /* V */
                size_t src_v = src_token_base + (size_t)(2*l+1) * ptl_bytes;
                size_t dst_v = (size_t)(2*l+1) * n_tokens * ptl_bytes
                             + (size_t)out_token * ptl_bytes;
                memcpy(dst + dst_v, svc->mmap_base + src_v, (size_t)ptl_bytes);
            }
        }
        copied += n;
    }
    return copied == n_tokens ? 0 : -1;
}

static int service_write(ds3_kv_cache_provider_t *p,
                         const char *session_id,
                         const int *tokens, int n_tokens,
                         const void *kv_buffer, size_t kv_buffer_bytes)
{
    ds3_kv_service_state_t *svc = (ds3_kv_service_state_t *)p->priv;
    if (service_ensure_mmap(svc) != 0) return -1;
    if (!kv_buffer || kv_buffer_bytes == 0) return -1;

    service_session_cache_t *sc = service_find_session_cache(svc, session_id);
    int cached_len = sc ? sc->cached_len : 0;
    if (cached_len < 0) cached_len = 0;
    if (cached_len > n_tokens) cached_len = n_tokens;

    int uncached = n_tokens - cached_len;
    if (uncached <= 0) return 0;

    int num_new_blocks = (uncached + svc->block_size - 1) / svc->block_size;
    if (num_new_blocks > DS3_KVC_SERVICE_MAX_BLOCKS) return -1;

    uint64_t hashes[DS3_KVC_SERVICE_MAX_BLOCKS];
    for (int i = 0; i < num_new_blocks; i++) {
        int start = cached_len + i * svc->block_size;
        int end = start + svc->block_size;
        if (end > n_tokens) end = n_tokens;
        hashes[i] = service_token_hash(&tokens[start], end - start);
    }

    ds3_kvcache_AllocateReq alloc_req = ds3_kvcache_AllocateReq_init_zero;
    alloc_req.session_id.funcs.encode = &encode_string;
    alloc_req.session_id.arg = (void *)session_id;
    alloc_req.num_blocks = num_new_blocks;
    encode_uint64_ctx_t hashes_ctx = { hashes, num_new_blocks, 0 };
    alloc_req.token_hashes.funcs.encode = &encode_repeated_uint64;
    alloc_req.token_hashes.arg = &hashes_ctx;

    service_block_t new_blocks[DS3_KVC_SERVICE_MAX_BLOCKS];
    decode_blocks_ctx_t alloc_blocks_ctx = { new_blocks, DS3_KVC_SERVICE_MAX_BLOCKS, 0 };

    ds3_kvcache_AllocateResp alloc_resp = ds3_kvcache_AllocateResp_init_zero;
    alloc_resp.blocks.funcs.decode = &decode_repeated_block_handle;
    alloc_resp.blocks.arg = &alloc_blocks_ctx;
    char err_buf[256] = {0};
    decode_string_ctx_t err_ctx = { err_buf, sizeof(err_buf) };
    alloc_resp.error.funcs.decode = &decode_string_to_buf;
    alloc_resp.error.arg = &err_ctx;

    if (service_call(svc, "Allocate",
                     ds3_kvcache_AllocateReq_fields, &alloc_req,
                     ds3_kvcache_AllocateResp_fields, &alloc_resp) != 0) {
        return -1;
    }
    if (!alloc_resp.ok) {
        ds3_log_warn("kv-cache Allocate failed: %s\n", err_buf[0] ? err_buf : "unknown");
        return -1;
    }
    if (alloc_blocks_ctx.n_blocks != num_new_blocks) return -1;

    const uint8_t *src = (const uint8_t *)kv_buffer;
    /* Host buffer is layer-interleaved:
     *   [L0_K(tok 0..P-1)][L0_V(tok 0..P-1)]...[L47_V(tok 0..P-1)]
     * mmap blocks are token-interleaved (matching Rust BlockStore layout):
     *   [tok0(all layers K+V)][tok1(all layers K+V)]...
     * so we transpose host -> block here. */
    int ptl_bytes = (int)svc->region_header->n_kv_head
                  * (int)svc->region_header->head_dim * (int)DS3_KVC_DEFAULT_DTYPE_SIZE;
    int n_layer = (int)svc->region_header->n_layer;

    for (int i = 0; i < num_new_blocks; i++) {
        int start = cached_len + i * svc->block_size;
        int end = start + svc->block_size;
        if (end > n_tokens) end = n_tokens;
        int chunk_tokens = end - start;

        for (int t = 0; t < chunk_tokens; t++) {
            int global_token = start + t;
            size_t dst_token_base = new_blocks[i].handle.mmap_offset
                                  + (size_t)t * svc->per_token_kv_bytes;
            for (int l = 0; l < n_layer; l++) {
                /* K */
                size_t src_k = (size_t)(2*l) * n_tokens * ptl_bytes
                             + (size_t)global_token * ptl_bytes;
                size_t dst_k = dst_token_base + (size_t)(2*l) * ptl_bytes;
                memcpy(svc->mmap_base + dst_k, src + src_k, (size_t)ptl_bytes);
                /* V */
                size_t src_v = (size_t)(2*l+1) * n_tokens * ptl_bytes
                             + (size_t)global_token * ptl_bytes;
                size_t dst_v = dst_token_base + (size_t)(2*l+1) * ptl_bytes;
                memcpy(svc->mmap_base + dst_v, src + src_v, (size_t)ptl_bytes);
            }
        }
    }

    /* Ensure all KV data written to mmap is visible to the Rust service
     * before we send the Store RPC.  Required by the protocol in §6.2. */
    __sync_synchronize();
    for (int i = 0; i < num_new_blocks; i++) {
        int start_tok = cached_len + i * svc->block_size;
        int end_tok = start_tok + svc->block_size;
        if (end_tok > n_tokens) end_tok = n_tokens;
        uint8_t *ptr = svc->mmap_base + new_blocks[i].handle.mmap_offset;
        size_t len = (size_t)(end_tok - start_tok) * svc->per_token_kv_bytes;
        msync(ptr, len, MS_SYNC);
    }

    uint64_t block_ids[DS3_KVC_SERVICE_MAX_BLOCKS];
    for (int i = 0; i < num_new_blocks; i++) {
        block_ids[i] = new_blocks[i].handle.block_id;
    }

    ds3_kvcache_StoreReq store_req = ds3_kvcache_StoreReq_init_zero;
    store_req.session_id.funcs.encode = &encode_string;
    store_req.session_id.arg = (void *)session_id;
    encode_int32_ctx_t tokens_ctx = { tokens, n_tokens, 0 };
    store_req.tokens.funcs.encode = &encode_repeated_int32;
    store_req.tokens.arg = &tokens_ctx;
    store_req.n_prompt = n_tokens;
    encode_uint64_ctx_t block_ids_ctx = { block_ids, num_new_blocks, 0 };
    store_req.block_ids.funcs.encode = &encode_repeated_uint64;
    store_req.block_ids.arg = &block_ids_ctx;
    store_req.cached_prefix_len = cached_len;

    ds3_kvcache_StatusResp store_resp = ds3_kvcache_StatusResp_init_zero;
    if (service_call(svc, "Store",
                     ds3_kvcache_StoreReq_fields, &store_req,
                     ds3_kvcache_StatusResp_fields, &store_resp) != 0) {
        return -1;
    }
    if (!store_resp.ok) return -1;

    if (sc) {
        sc->cached_len = n_tokens;
        sc->n_blocks = 0;
    }
    return 0;
}

const struct ds3_kv_cache_provider_vtable ds3_kv_service_provider = {
    .name = "service",
    .open = service_open,
    .close = service_close,
    .lookup = service_lookup,
    .read = service_read,
    .write = service_write,
    .create_session = service_create_session,
    .close_session = service_close_session,
    .reset_session = service_reset_session,
};
