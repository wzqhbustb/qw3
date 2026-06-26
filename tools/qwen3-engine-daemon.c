/*
 * tools/qwen3-engine-daemon.c — Resident Qwen3 inference daemon
 *
 * Phase 1 implementation:
 *   - Loads the model once and keeps it resident in memory.
 *   - Speaks NDJSON over a Unix Domain Socket.
 *   - Maintains multiple sessions with independent conversation state.
 *
 * Limitations:
 *   - Requests are processed serially (Phase 1).
 *   - The underlying ds3_engine_generate_ex resets the KV cache each call, so
 *     the daemon rebuilds the full prompt on every generate.  True incremental
 *     KV-cache reuse requires engine support (Phase 3 / KV Cache Service).
 */

#include "src/ds3.h"

#define ERR_CONTEXT_OVERFLOW -32003
#define ERR_GENERATION_TIMEOUT -32004
#define ERR_REQUEST_TOO_LARGE  -32006

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

/* ============================================================================
 * Buffer helpers (adapted from ds4/ds4_server.c)
 * ============================================================================ */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf_t;

static void buf_reserve(buf_t *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) {
        fprintf(stderr, "buffer overflow\n");
        exit(1);
    }
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    b->ptr = (char *)realloc(b->ptr, cap);
    if (!b->ptr) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    b->cap = cap;
}

static void buf_append(buf_t *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    if (b->ptr) b->ptr[b->len] = '\0';
}

static void buf_putc(buf_t *b, char c) { buf_append(b, &c, 1); }
static void buf_puts(buf_t *b, const char *s) { buf_append(b, s, strlen(s)); }

static void buf_printf(buf_t *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        fprintf(stderr, "vsnprintf failed\n");
        exit(1);
    }
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *buf_take(buf_t *b) {
    char *p = b->ptr ? b->ptr : strdup("");
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
    return p;
}

static void buf_free(buf_t *b) {
    free(b->ptr);
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
}

/* ============================================================================
 * Minimal JSON parser (adapted from ds4/ds4_server.c)
 * Only understands the request/response shapes used by this daemon.
 * ============================================================================ */

#define JSON_MAX_NESTING 256

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(buf_t *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static bool json_string(const char **p, char **out) {
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf_t b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v < 0) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) { *out = true; return true; }
    if (json_lit(p, "false")) { *out = false; return true; }
    return false;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') { (*p)++; return true; }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') { (*p)++; return true; }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) { return json_skip_value_depth(p, 0); }

/* ============================================================================
 * Data structures
 * ============================================================================ */

typedef struct {
    char *role;        /* "user" | "assistant" | "tool_result" | "system" */
    char *content;
    char *tool_calls;  /* JSON array string, for assistant messages */
    char *tool_call_id;
    char *tool_name;
} msg_t;

typedef struct session {
    char *id;
    char *system_prompt;
    float temperature;
    int max_tokens;
    int n_ctx;

    msg_t *messages;
    int n_messages;
    int msg_capacity;

    int total_generated;
    time_t last_active;

    /* Provider-side session has been created (so we don't recreate it every
     * generate call). */
    int provider_session_created;
} session_t;

typedef struct {
    char *method;
    char *id;
    char *params;      /* raw JSON object text */
} request_t;

typedef struct {
    ds3_engine_t *engine;
    ds3_kv_cache_provider_t *kv_provider;
    session_t **sessions;
    int n_sessions;
    int session_capacity;
    int n_ctx;
    int total_generated;
    int total_prompt_tokens;
    size_t max_request_size;
    double generate_timeout;
    time_t started_at;
} daemon_state_t;

/* ============================================================================
 * Request parsing helpers
 * ============================================================================ */

static void request_free(request_t *r) {
    free(r->method);
    free(r->id);
    free(r->params);
    r->method = NULL;
    r->id = NULL;
    r->params = NULL;
}

static bool parse_request(const char *json, request_t *out) {
    memset(out, 0, sizeof(*out));
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') { p++; break; }
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') { free(key); return false; }
        p++;
        if (strcmp(key, "method") == 0) {
            free(out->method);
            if (!json_string(&p, &out->method)) { free(key); return false; }
        } else if (strcmp(key, "id") == 0) {
            free(out->id);
            if (*p == '"') {
                if (!json_string(&p, &out->id)) { free(key); return false; }
            } else {
                double v = 0;
                if (!json_number(&p, &v)) { free(key); return false; }
                out->id = (char *)malloc(32);
                snprintf(out->id, 32, "%.0f", v);
            }
        } else if (strcmp(key, "params") == 0) {
            free(out->params);
            const char *start = p;
            if (!json_skip_value(&p)) { free(key); return false; }
            size_t n = (size_t)(p - start);
            out->params = (char *)malloc(n + 1);
            memcpy(out->params, start, n);
            out->params[n] = '\0';
        } else {
            if (!json_skip_value(&p)) { free(key); return false; }
        }
        free(key);
        json_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        return false;
    }
    return out->method != NULL;
}

static bool params_get_string(const char *params, const char *key, char **out) {
    *out = NULL;
    const char *p = params;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') return false;
        char *k = NULL;
        if (!json_string(&p, &k)) return false;
        json_ws(&p);
        if (*p != ':') { free(k); return false; }
        p++;
        bool match = strcmp(k, key) == 0;
        free(k);
        if (match) {
            if (*p == '"') {
                return json_string(&p, out);
            }
            return false;
        }
        if (!json_skip_value(&p)) return false;
        json_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return false;
        return false;
    }
}

static bool params_get_int(const char *params, const char *key, int *out) {
    const char *p = params;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') return false;
        char *k = NULL;
        if (!json_string(&p, &k)) return false;
        json_ws(&p);
        if (*p != ':') { free(k); return false; }
        p++;
        bool match = strcmp(k, key) == 0;
        free(k);
        if (match) {
            return json_int(&p, out);
        }
        if (!json_skip_value(&p)) return false;
        json_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return false;
        return false;
    }
}

static bool params_get_float(const char *params, const char *key, float *out) {
    double v = 0.0;
    const char *p = params;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') return false;
        char *k = NULL;
        if (!json_string(&p, &k)) return false;
        json_ws(&p);
        if (*p != ':') { free(k); return false; }
        p++;
        bool match = strcmp(k, key) == 0;
        free(k);
        if (match) {
            if (!json_number(&p, &v)) return false;
            *out = (float)v;
            return true;
        }
        if (!json_skip_value(&p)) return false;
        json_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return false;
        return false;
    }
}

static bool params_get_bool(const char *params, const char *key, bool *out, bool default_val) {
    *out = default_val;
    const char *p = params;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') return true;
        char *k = NULL;
        if (!json_string(&p, &k)) return false;
        json_ws(&p);
        if (*p != ':') { free(k); return false; }
        p++;
        bool match = strcmp(k, key) == 0;
        free(k);
        if (match) {
            return json_bool(&p, out);
        }
        if (!json_skip_value(&p)) return false;
        json_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return true;
        return false;
    }
}

/* ============================================================================
 * Session management
 * ============================================================================ */

static void msg_free(msg_t *m) {
    free(m->role);
    free(m->content);
    free(m->tool_calls);
    free(m->tool_call_id);
    free(m->tool_name);
    memset(m, 0, sizeof(*m));
}

static void session_free(session_t *s) {
    if (!s) return;
    free(s->id);
    free(s->system_prompt);
    for (int i = 0; i < s->n_messages; i++) msg_free(&s->messages[i]);
    free(s->messages);
    free(s);
}

static session_t *session_find(daemon_state_t *ds, const char *id) {
    for (int i = 0; i < ds->n_sessions; i++) {
        if (strcmp(ds->sessions[i]->id, id) == 0) return ds->sessions[i];
    }
    return NULL;
}

static void session_close(daemon_state_t *ds, const char *id) {
    for (int i = 0; i < ds->n_sessions; i++) {
        if (strcmp(ds->sessions[i]->id, id) == 0) {
            if (ds->kv_provider && ds->sessions[i]->provider_session_created) {
                ds3_kv_cache_close_session(ds->kv_provider, id);
            }
            session_free(ds->sessions[i]);
            ds->sessions[i] = ds->sessions[--ds->n_sessions];
            return;
        }
    }
}

static session_t *session_create(daemon_state_t *ds, const char *id,
                                 const char *system_prompt,
                                 float temperature, int max_tokens) {
    session_close(ds, id); /* remove any existing session to avoid dangling pointers */
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    if (!s) return NULL;
    s->id = strdup(id);
    s->system_prompt = strdup(system_prompt ? system_prompt : "");
    s->temperature = temperature > 0 ? temperature : 0.7f;
    s->max_tokens = max_tokens > 0 ? max_tokens : 2048;
    s->n_ctx = ds->n_ctx;
    s->msg_capacity = 16;
    s->messages = (msg_t *)calloc((size_t)s->msg_capacity, sizeof(msg_t));
    if (!s->id || !s->system_prompt || !s->messages) {
        session_free(s);
        return NULL;
    }
    s->last_active = time(NULL);
    if (ds->n_sessions >= ds->session_capacity) {
        int new_cap = ds->session_capacity ? ds->session_capacity * 2 : 8;
        session_t **tmp = (session_t **)realloc(ds->sessions, (size_t)new_cap * sizeof(session_t *));
        if (!tmp) {
            session_free(s);
            return NULL;
        }
        ds->sessions = tmp;
        ds->session_capacity = new_cap;
    }
    ds->sessions[ds->n_sessions++] = s;
    return s;
}

static void session_append_message(session_t *s, const char *role,
                                   const char *content,
                                   const char *tool_calls,
                                   const char *tool_call_id,
                                   const char *tool_name) {
    if (s->n_messages >= s->msg_capacity) {
        int new_cap = s->msg_capacity * 2;
        s->messages = (msg_t *)realloc(s->messages, (size_t)new_cap * sizeof(msg_t));
        memset(&s->messages[s->msg_capacity], 0, (size_t)(new_cap - s->msg_capacity) * sizeof(msg_t));
        s->msg_capacity = new_cap;
    }
    msg_t *m = &s->messages[s->n_messages++];
    m->role = strdup(role ? role : "");
    m->content = strdup(content ? content : "");
    m->tool_calls = tool_calls ? strdup(tool_calls) : NULL;
    m->tool_call_id = tool_call_id ? strdup(tool_call_id) : NULL;
    m->tool_name = tool_name ? strdup(tool_name) : NULL;
    s->last_active = time(NULL);
}

/* Replace the in-memory conversation history.  KV-cache preservation is decided
 * by the caller (handle_replace_history) before/after this function runs. */
static void session_replace_history(session_t *s, const msg_t *msgs, int n_msgs) {
    for (int i = 0; i < s->n_messages; i++) msg_free(&s->messages[i]);
    s->n_messages = 0;
    for (int i = 0; i < n_msgs; i++) {
        session_append_message(s, msgs[i].role, msgs[i].content,
                               msgs[i].tool_calls, msgs[i].tool_call_id,
                               msgs[i].tool_name);
    }
    s->last_active = time(NULL);
}

/* ============================================================================
 * Chat template formatting (must match agent/agent.go writeMessage())
 * ============================================================================ */

static void json_escape_to_buf(buf_t *b, const char *s) {
    if (!s) return;
    while (*s) {
        unsigned char c = (unsigned char)*s++;
        switch (c) {
        case '"': buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        default:
            if (c < 0x20) {
                buf_printf(b, "\\u%04x", c);
            } else {
                buf_putc(b, (char)c);
            }
        }
    }
}

/* Append a single message to the chat prompt using the same template as
 * agent.go:writeMessage().  The output is plain text containing
 * <|im_start|>/role/<|im_end|> markers; ds3_engine_tokenize() maps these
 * strings to the correct special-token IDs. */
static void append_chat_message(buf_t *b, const msg_t *m) {
    if (strcmp(m->role, "user") == 0) {
        buf_puts(b, "<|im_start|>user\n");
        buf_puts(b, m->content ? m->content : "");
        buf_puts(b, "<|im_end|>\n");
    } else if (strcmp(m->role, "assistant") == 0) {
        buf_puts(b, "<|im_start|>assistant\n");
        buf_puts(b, m->content ? m->content : "");
        if (m->tool_calls && m->tool_calls[0]) {
            buf_puts(b, "\n<tool_call>\n");
            buf_puts(b, m->tool_calls);
            buf_puts(b, "\n</tool_call>\n");
        }
        buf_puts(b, "<|im_end|>\n");
    } else if (strcmp(m->role, "tool_result") == 0) {
        buf_puts(b, "<|im_start|>tool\n");
        if (m->tool_call_id && m->tool_name) {
            buf_printf(b, "[%s] %s: %s",
                       m->tool_call_id, m->tool_name,
                       m->content ? m->content : "");
        } else {
            buf_puts(b, m->content ? m->content : "");
        }
        buf_puts(b, "<|im_end|>\n");
    } else if (strcmp(m->role, "system") == 0) {
        buf_puts(b, "<|im_start|>system\n");
        buf_puts(b, m->content ? m->content : "");
        buf_puts(b, "<|im_end|>\n");
    }
}

/* Build the full multi-turn Qwen3 chat prompt, byte-for-byte compatible with
 * agent.go:buildPrompt().  The returned string is tokenized directly and
 * passed to ds3_engine_generate_ex(). */
static int build_chat_prompt(session_t *s, const char *user_prompt,
                             const char *context, buf_t *out) {
    buf_reserve(out, 4096);

    /* System prompt, optionally extended with per-turn context (memory notes). */
    buf_puts(out, "<|im_start|>system\n");
    buf_puts(out, s->system_prompt ? s->system_prompt : "");
    if (context && context[0]) {
        buf_puts(out, "\n\n");
        buf_puts(out, context);
    }
    buf_puts(out, "<|im_end|>\n");

    /* Conversation history. */
    for (int i = 0; i < s->n_messages; i++) {
        append_chat_message(out, &s->messages[i]);
    }

    /* Current user turn (may be empty for continue-after-tool-result). */
    if (user_prompt && user_prompt[0]) {
        buf_puts(out, "<|im_start|>user\n");
        buf_puts(out, user_prompt);
        buf_puts(out, "<|im_end|>\n");
    }

    /* Assistant prefix ready for generation.
     * Append the official Qwen3 no-think empty <think> block so the model
     * closes any reasoning quickly and emits the final answer in the same
     * generation call.  Any <think>...</think> content is stripped before the
     * text is returned or stored in history. */
    buf_puts(out, "<|im_start|>assistant\n");
    buf_puts(out, "<think>\n\n</think>\n\n");
    return 0;
}

/* Remove Qwen3 <think>...</think> reasoning blocks from generated text and
 * trim leading whitespace.  The no-think prefix leaves the opening <think>
 * implicit, so we strip up to and including the first </think> tag and any
 * following whitespace.  The caller must free the returned string. */
static char *strip_think_blocks(const char *text) {
    if (!text) return NULL;
    const char *think_start = strstr(text, "<think>");
    const char *think_end   = strstr(text, "</think>");
    const char *answer = text;
    if (think_end) {
        answer = think_end + strlen("</think>");
    } else if (think_start) {
        /* Unclosed think block: drop from the opening tag onward. */
        answer = think_start + strlen("<think>");
    }
    /* Swallow trailing whitespace after the reasoning block. */
    while (*answer == '\r' || *answer == '\n') answer++;
    /* Trim leading whitespace of the final answer. */
    while (*answer == ' ' || *answer == '\t') answer++;
    return strdup(answer);
}

/* Extract structured tool calls from assistant text.  Supports JSON objects or
 * arrays inside <tool_call>...</tool_call> tags.  Returns a newly allocated
 * JSON array string (e.g. "[{\"tool\":\"read\"}]"), or NULL if no valid JSON
 * tool calls are found.  The caller must free the returned string. */
static char *extract_tool_calls_json(const char *text) {
    if (!text) return NULL;
    buf_t out = {0};
    buf_puts(&out, "[");
    bool first = true;
    const char *p = text;
    for (;;) {
        const char *start = strstr(p, "<tool_call>");
        if (!start) break;
        const char *end = strstr(start, "</tool_call>");
        if (!end) break;
        const char *body = start + strlen("<tool_call>");
        while (body < end && isspace((unsigned char)*body)) body++;
        const char *body_end = end;
        while (body_end > body && isspace((unsigned char)body_end[-1])) body_end--;
        size_t len = (size_t)(body_end - body);
        if (len == 0) {
            p = end + strlen("</tool_call>");
            continue;
        }
        char *item = (char *)malloc(len + 1);
        memcpy(item, body, len);
        item[len] = '\0';

        /* Validate that the body looks like JSON (object or array). */
        bool ok = false;
        if (item[0] == '{' || item[0] == '[') {
            /* Simple structural check: braces/brackets must balance. */
            int depth = 0;
            bool in_string = false;
            ok = true;
            for (size_t i = 0; i < len; i++) {
                if (item[i] == '"' && (i == 0 || item[i - 1] != '\\')) {
                    in_string = !in_string;
                } else if (!in_string) {
                    if (item[i] == '{' || item[i] == '[') depth++;
                    else if (item[i] == '}' || item[i] == ']') depth--;
                    if (depth < 0) { ok = false; break; }
                }
            }
            if (ok && depth != 0) ok = false;
        }

        if (ok) {
            if (!first) buf_puts(&out, ",");
            first = false;
            if (item[0] == '[') {
                /* Inline array contents without surrounding brackets. */
                buf_append(&out, item + 1, len - 2);
            } else {
                buf_puts(&out, item);
            }
        }
        free(item);
        p = end + strlen("</tool_call>");
    }
    buf_puts(&out, "]");
    if (first) {
        buf_free(&out);
        return NULL;
    }
    return buf_take(&out);
}

/* ============================================================================
 * Global server state
 * ============================================================================ */

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_active_connections = 0;
static volatile sig_atomic_t g_active_generate = 0;

/* ============================================================================
 * Generation
 * ============================================================================ */

static int generate_for_session(daemon_state_t *ds, session_t *s,
                                const char *user_prompt, const char *context,
                                float temperature, int max_tokens,
                                char **out_text, int *out_prompt_tokens,
                                int *out_gen_tokens,
                                int *out_cached_prefix_len,
                                int *out_new_cached_prefix_len) {
    buf_t prompt_buf = {0};
    build_chat_prompt(s, user_prompt, context, &prompt_buf);

    int max_prompt = s->n_ctx;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt * sizeof(int));
    if (!prompt_tokens) {
        buf_free(&prompt_buf);
        return -1;
    }
    /* Tokenize the full multi-turn prompt.  The <|im_start|>/role/<|im_end|>
     * strings are mapped by the tokenizer to the same special-token IDs used
     * by ds3_engine_chat_format, so the result matches agent.go:buildPrompt(). */
    int n_prompt = ds3_tokenize(ds->engine, prompt_buf.ptr,
                                prompt_tokens, max_prompt);
    buf_free(&prompt_buf);
    if (n_prompt < 0) {
        free(prompt_tokens);
        return -1;
    }
    /* ds3_tokenize silently truncates to max_prompt.  If we got back the
     * entire buffer the prompt is at least n_ctx tokens long and there is no
     * room left for generation.  Report this as a context overflow so the
     * client can compact or reset instead of treating it as a generic failure. */
    if (n_prompt >= max_prompt) {
        free(prompt_tokens);
        return -2;
    }

    int max_gen = max_tokens > 0 ? max_tokens : s->max_tokens;
    int *output_tokens = (int *)malloc((size_t)max_gen * sizeof(int));
    if (!output_tokens) {
        free(prompt_tokens);
        return -1;
    }

    float temp = temperature >= 0 ? temperature : s->temperature;

    /* Tell the engine which session owns the KV cache for this generate. */
    ds3_engine_set_session_id(ds->engine, s->id);
    if (ds->kv_provider && !s->provider_session_created) {
        ds3_kv_cache_create_session(ds->kv_provider, s->id);
        s->provider_session_created = 1;
    }

    g_active_generate++;
    ds3_engine_set_generate_deadline(ds->engine, ds->generate_timeout);
    int n_gen = ds3_engine_generate_ex(ds->engine, prompt_tokens, n_prompt,
                                       max_gen, temp,
                                       output_tokens, max_gen, NULL, NULL);
    ds3_engine_clear_generate_deadline(ds->engine);
    g_active_generate--;
    free(prompt_tokens);
    if (n_gen == -2) {
        free(output_tokens);
        return -3;
    }
    if (n_gen < 0) {
        free(output_tokens);
        return -1;
    }

    size_t text_cap = (size_t)n_gen * 64 + 1;
    char *text = (char *)malloc(text_cap);
    if (!text) {
        free(output_tokens);
        return -1;
    }
    int bytes = ds3_engine_decode_sequence(ds->engine, output_tokens, n_gen,
                                           text, text_cap);
    if (bytes < 0) bytes = 0;
    text[bytes] = '\0';
    free(output_tokens);

    char *stripped = strip_think_blocks(text);
    free(text);
    if (!stripped) return -1;

    *out_text = stripped;
    *out_prompt_tokens = n_prompt;
    *out_gen_tokens = n_gen;
    *out_cached_prefix_len = ds3_engine_last_cached_prefix_len(ds->engine);
    *out_new_cached_prefix_len = ds3_engine_last_new_cached_prefix_len(ds->engine);
    s->total_generated += n_gen;
    ds->total_generated += n_gen;
    ds->total_prompt_tokens += n_prompt;
    s->last_active = time(NULL);
    return 0;
}

/* ============================================================================
 * Parsing structured messages (for replace_history)
 * ============================================================================ */

static bool parse_message_array(const char *json, msg_t **out_msgs, int *out_n) {
    *out_msgs = NULL;
    *out_n = 0;
    const char *p = json;
    json_ws(&p);
    if (*p != '[') return false;
    p++;
    json_ws(&p);
    if (*p == ']') { p++; return true; }

    int cap = 8;
    msg_t *msgs = (msg_t *)calloc((size_t)cap, sizeof(msg_t));
    int n = 0;

    for (;;) {
        json_ws(&p);
        if (*p != '{') goto fail;
        p++;
        if (n >= cap) {
            cap *= 2;
            msgs = (msg_t *)realloc(msgs, (size_t)cap * sizeof(msg_t));
        }
        msg_t *m = &msgs[n];
        memset(m, 0, sizeof(*m));
        for (;;) {
            json_ws(&p);
            if (*p == '}') { p++; break; }
            char *key = NULL;
            if (!json_string(&p, &key)) goto fail;
            json_ws(&p);
            if (*p != ':') { free(key); goto fail; }
            p++;
            if (strcmp(key, "role") == 0) {
                free(m->role);
                json_string(&p, &m->role);
            } else if (strcmp(key, "content") == 0) {
                free(m->content);
                json_string(&p, &m->content);
            } else if (strcmp(key, "tool_calls") == 0) {
                free(m->tool_calls);
                const char *start = p;
                if (!json_skip_value(&p)) { free(key); goto fail; }
                size_t len = (size_t)(p - start);
                m->tool_calls = (char *)malloc(len + 1);
                memcpy(m->tool_calls, start, len);
                m->tool_calls[len] = '\0';
            } else if (strcmp(key, "tool_call_id") == 0) {
                free(m->tool_call_id);
                json_string(&p, &m->tool_call_id);
            } else if (strcmp(key, "tool_name") == 0) {
                free(m->tool_name);
                json_string(&p, &m->tool_name);
            } else {
                if (!json_skip_value(&p)) { free(key); goto fail; }
            }
            free(key);
            json_ws(&p);
            if (*p == ',') { p++; continue; }
            if (*p == '}') { p++; break; }
            goto fail;
        }
        n++;
        json_ws(&p);
        if (*p == ']') { p++; break; }
        if (*p == ',') { p++; continue; }
        goto fail;
    }
    *out_msgs = msgs;
    *out_n = n;
    return true;
fail:
    for (int i = 0; i < n; i++) msg_free(&msgs[i]);
    free(msgs);
    return false;
}

/* ============================================================================
 * JSON response helpers
 * ============================================================================ */

static void send_response(int fd, const char *id, bool ok,
                          const char *result_json,
                          int err_code, const char *err_msg) {
    buf_t b = {0};
    buf_puts(&b, "{");
    if (id && id[0]) {
        buf_puts(&b, "\"id\":\"");
        json_escape_to_buf(&b, id);
        buf_puts(&b, "\",");
    }
    if (ok) {
        buf_puts(&b, "\"result\":");
        buf_puts(&b, result_json ? result_json : "{}");
    } else {
        buf_printf(&b, "\"error\":{\"code\":%d,\"message\":\"", err_code);
        json_escape_to_buf(&b, err_msg);
        buf_puts(&b, "\"}");
    }
    buf_puts(&b, "}\n");
    size_t off = 0;
    while (off < b.len) {
        ssize_t n = write(fd, b.ptr + off, b.len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        off += (size_t)n;
    }
    buf_free(&b);
}

static void send_error(int fd, const char *id, int code, const char *msg) {
    send_response(fd, id, false, NULL, code, msg);
}

/* ============================================================================
 * Request handlers
 * ============================================================================ */

static void handle_load_model(daemon_state_t *ds, const request_t *req, int fd) {
    char *path = NULL;
    int n_ctx = ds->n_ctx;
    if (!params_get_string(req->params, "model_path", &path) || !path || !path[0]) {
        send_error(fd, req->id, -32602, "missing model_path");
        return;
    }
    params_get_int(req->params, "n_ctx", &n_ctx);
    if (n_ctx <= 0) n_ctx = ds->n_ctx;

    if (ds->engine) {
        /* Close existing sessions; engine will be reopened. */
        for (int i = 0; i < ds->n_sessions; i++) {
            if (ds->kv_provider && ds->sessions[i]->provider_session_created) {
                ds3_kv_cache_close_session(ds->kv_provider, ds->sessions[i]->id);
            }
            session_free(ds->sessions[i]);
        }
        ds->n_sessions = 0;
        ds3_engine_close(ds->engine);
        ds->engine = NULL;
    }

    ds->engine = ds3_engine_open(path, n_ctx);
    free(path);
    if (!ds->engine) {
        send_error(fd, req->id, -32001, "failed to load model");
        return;
    }
    if (ds->kv_provider) {
        ds3_engine_set_kv_provider(ds->engine, ds->kv_provider);
    }
    ds->n_ctx = n_ctx;

    buf_t b = {0};
    buf_printf(&b, "{\"loaded\":true,\"model_name\":\"%s\",\"n_ctx\":%d,\"vocab_size\":%d}",
               DS3_MODEL_NAME, n_ctx, DS3_N_VOCAB);
    send_response(fd, req->id, true, b.ptr, 0, NULL);
    buf_free(&b);
}

static void handle_create_session(daemon_state_t *ds, const request_t *req, int fd) {
    char *id = NULL;
    char *system_prompt = NULL;
    float temperature = 0.7f;
    int max_tokens = 2048;
    if (!params_get_string(req->params, "session_id", &id) || !id || !id[0]) {
        send_error(fd, req->id, -32602, "missing session_id");
        return;
    }
    params_get_string(req->params, "system_prompt", &system_prompt);
    params_get_float(req->params, "temperature", &temperature);
    params_get_int(req->params, "max_tokens", &max_tokens);

    session_t *s = session_create(ds, id, system_prompt, temperature, max_tokens);
    if (!s) {
        free(id);
        free(system_prompt);
        send_error(fd, req->id, -32005, "failed to create session");
        return;
    }

    if (ds->kv_provider) {
        ds3_kv_cache_create_session(ds->kv_provider, id);
        s->provider_session_created = 1;
    }

    buf_t b = {0};
    buf_puts(&b, "{\"session_id\":\"");
    json_escape_to_buf(&b, id);
    buf_puts(&b, "\",\"created_at\":\"");
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", tm_info);
    buf_puts(&b, tbuf);
    buf_puts(&b, "\"}");
    send_response(fd, req->id, true, b.ptr, 0, NULL);
    buf_free(&b);

    free(id);
    free(system_prompt);
}

static void handle_generate(daemon_state_t *ds, const request_t *req, int fd) {
    if (!ds->engine) {
        send_error(fd, req->id, -32001, "model not loaded");
        return;
    }
    char *session_id = NULL;
    char *user_prompt = NULL;
    char *context = NULL;
    if (!params_get_string(req->params, "session_id", &session_id) || !session_id || !session_id[0]) {
        send_error(fd, req->id, -32602, "missing session_id");
        return;
    }
    session_t *s = session_find(ds, session_id);
    if (!s) {
        free(session_id);
        send_error(fd, req->id, -32002, "session not found");
        return;
    }
    params_get_string(req->params, "user_prompt", &user_prompt);
    params_get_string(req->params, "context", &context);
    float req_temp = -1.0f;
    int req_max = 0;
    params_get_float(req->params, "temperature", &req_temp);
    params_get_int(req->params, "max_tokens", &req_max);

    bool has_user = user_prompt && user_prompt[0];

    char *text = NULL;
    int n_prompt = 0, n_gen = 0;
    int cached_prefix_len = 0, new_cached_prefix_len = 0;
    int rc = generate_for_session(ds, s, has_user ? user_prompt : "", context,
                                  req_temp, req_max, &text, &n_prompt, &n_gen,
                                  &cached_prefix_len, &new_cached_prefix_len);
    free(context);
    if (rc == -2) {
        free(session_id);
        free(user_prompt);
        free(text);
        send_error(fd, req->id, ERR_CONTEXT_OVERFLOW, "context overflow");
        return;
    }
    if (rc == -3) {
        free(session_id);
        free(user_prompt);
        free(text);
        send_error(fd, req->id, ERR_GENERATION_TIMEOUT, "generation timeout");
        return;
    }
    if (rc != 0) {
        free(session_id);
        free(user_prompt);
        free(text);
        send_error(fd, req->id, -32005, "generation failed");
        return;
    }

    /* Append user and assistant turns only after successful generation. */
    if (has_user) {
        session_append_message(s, "user", user_prompt, NULL, NULL, NULL);
    }
    free(user_prompt);
    session_append_message(s, "assistant", text, NULL, NULL, NULL);

    int current_tokens = n_prompt + n_gen;
    char *tool_calls_json = extract_tool_calls_json(text);
    buf_t b = {0};
    buf_puts(&b, "{\"session_id\":\"");
    json_escape_to_buf(&b, session_id);
    buf_puts(&b, "\",\"text\":\"");
    json_escape_to_buf(&b, text);
    buf_printf(&b, "\",\"tokens_generated\":%d,\"tokens_prompt\":%d,\"current_tokens\":%d,\"max_tokens\":%d,\"n_ctx\":%d,\"cached_prefix_len\":%d,\"new_cached_prefix_len\":%d,\"tool_calls\":",
               n_gen, n_prompt, current_tokens, s->max_tokens, s->n_ctx,
               cached_prefix_len, new_cached_prefix_len);
    buf_puts(&b, tool_calls_json ? tool_calls_json : "[]");
    buf_puts(&b, "}");
    send_response(fd, req->id, true, b.ptr, 0, NULL);
    buf_free(&b);
    free(tool_calls_json);
    free(text);
    free(session_id);
}

static void handle_append_turn(daemon_state_t *ds, const request_t *req, int fd) {
    char *session_id = NULL;
    char *role = NULL;
    char *content = NULL;
    char *tool_call_id = NULL;
    char *tool_name = NULL;
    if (!params_get_string(req->params, "session_id", &session_id) || !session_id || !session_id[0]) {
        send_error(fd, req->id, -32602, "missing session_id");
        return;
    }
    session_t *s = session_find(ds, session_id);
    if (!s) {
        free(session_id);
        send_error(fd, req->id, -32002, "session not found");
        return;
    }
    if (!params_get_string(req->params, "role", &role) || !role || !role[0]) {
        free(session_id);
        send_error(fd, req->id, -32602, "missing role");
        return;
    }
    params_get_string(req->params, "content", &content);
    params_get_string(req->params, "tool_call_id", &tool_call_id);
    params_get_string(req->params, "tool_name", &tool_name);

    session_append_message(s, role, content ? content : "", NULL, tool_call_id, tool_name);

    free(session_id);
    free(role);
    free(content);
    free(tool_call_id);
    free(tool_name);

    send_response(fd, req->id, true, "{\"appended\":true}", 0, NULL);
}

static void handle_replace_history(daemon_state_t *ds, const request_t *req, int fd) {
    char *session_id = NULL;
    char *messages_json = NULL;
    if (!params_get_string(req->params, "session_id", &session_id) || !session_id || !session_id[0]) {
        send_error(fd, req->id, -32602, "missing session_id");
        return;
    }
    session_t *s = session_find(ds, session_id);
    if (!s) {
        free(session_id);
        send_error(fd, req->id, -32002, "session not found");
        return;
    }
    bool preserve = false;
    params_get_bool(req->params, "preserve_kv_cache", &preserve, false);

    /* Extract raw messages array from params. */
    const char *p = req->params;
    json_ws(&p);
    if (*p == '{') p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') break;
        char *k = NULL;
        if (!json_string(&p, &k)) break;
        json_ws(&p);
        if (*p != ':') { free(k); break; }
        p++;
        if (strcmp(k, "messages") == 0) {
            free(k);
            const char *start = p;
            if (!json_skip_value(&p)) break;
            size_t n = (size_t)(p - start);
            messages_json = (char *)malloc(n + 1);
            memcpy(messages_json, start, n);
            messages_json[n] = '\0';
            break;
        }
        if (!json_skip_value(&p)) break;
        json_ws(&p);
        if (*p == ',') p++;
    }

    if (!messages_json) {
        free(session_id);
        send_error(fd, req->id, -32602, "missing messages");
        return;
    }

    msg_t *msgs = NULL;
    int n_msgs = 0;
    if (!parse_message_array(messages_json, &msgs, &n_msgs)) {
        free(messages_json);
        free(session_id);
        send_error(fd, req->id, -32602, "invalid messages");
        return;
    }
    free(messages_json);

    session_replace_history(s, msgs, n_msgs);
    for (int i = 0; i < n_msgs; i++) msg_free(&msgs[i]);
    free(msgs);

    /* Invalidate KV cache when history is replaced without preserving.
     * The token sequence has changed so any cached prefix is invalid.
     *
     * preserve=true is only safe when the new history's token prefix matches
     * the old one (e.g. after append_turn).  Full history replacement, such as
     * compaction, must reset the cache to avoid using stale logits. */
    if (!preserve && ds->kv_provider) {
        ds3_kv_cache_reset_session(ds->kv_provider, session_id);
    }

    free(session_id);
    send_response(fd, req->id, true, "{\"replaced\":true}", 0, NULL);
}

static void handle_reset_session(daemon_state_t *ds, const request_t *req, int fd) {
    char *session_id = NULL;
    if (!params_get_string(req->params, "session_id", &session_id) || !session_id || !session_id[0]) {
        send_error(fd, req->id, -32602, "missing session_id");
        return;
    }
    session_t *s = session_find(ds, session_id);
    if (!s) {
        free(session_id);
        send_error(fd, req->id, -32002, "session not found");
        return;
    }
    for (int i = 0; i < s->n_messages; i++) msg_free(&s->messages[i]);
    s->n_messages = 0;
    if (ds->kv_provider && s->provider_session_created) {
        ds3_kv_cache_reset_session(ds->kv_provider, session_id);
    }
    free(session_id);
    send_response(fd, req->id, true, "{\"reset\":true}", 0, NULL);
}

static void handle_close_session(daemon_state_t *ds, const request_t *req, int fd) {
    char *session_id = NULL;
    if (!params_get_string(req->params, "session_id", &session_id) || !session_id || !session_id[0]) {
        send_error(fd, req->id, -32602, "missing session_id");
        return;
    }
    session_close(ds, session_id);
    free(session_id);
    send_response(fd, req->id, true, "{\"closed\":true}", 0, NULL);
}

static void handle_health(daemon_state_t *ds, const request_t *req, int fd) {
    buf_t b = {0};
    buf_printf(&b, "{\"status\":\"ok\",\"model_loaded\":%s,\"active_sessions\":%d,\"total_tokens_generated\":%d,\"uptime_seconds\":%ld}",
               ds->engine ? "true" : "false", ds->n_sessions, ds->total_generated,
               (long)(time(NULL) - ds->started_at));
    send_response(fd, req->id, true, b.ptr, 0, NULL);
    buf_free(&b);
}

static void handle_stats(daemon_state_t *ds, const request_t *req, int fd) {
    buf_t b = {0};
    buf_printf(&b, "{"
               "\"status\":\"ok\","
               "\"model_name\":\"%s\","
               "\"model_loaded\":%s,"
               "\"n_ctx\":%d,"
               "\"vocab_size\":%d,"
               "\"gpu_layers\":%d,"
               "\"active_sessions\":%d,"
               "\"total_tokens_generated\":%d,"
               "\"total_tokens_prompt\":%d,"
               "\"uptime_seconds\":%ld"
               "}",
               DS3_MODEL_NAME,
               ds->engine ? "true" : "false",
               ds->n_ctx,
               DS3_N_VOCAB,
               DS3_N_LAYER,
               ds->n_sessions,
               ds->total_generated,
               ds->total_prompt_tokens,
               (long)(time(NULL) - ds->started_at));
    send_response(fd, req->id, true, b.ptr, 0, NULL);
    buf_free(&b);
}

static void dispatch_request(daemon_state_t *ds, const request_t *req, int fd) {
    if (strcmp(req->method, "load_model") == 0) handle_load_model(ds, req, fd);
    else if (strcmp(req->method, "create_session") == 0) handle_create_session(ds, req, fd);
    else if (strcmp(req->method, "generate") == 0) handle_generate(ds, req, fd);
    else if (strcmp(req->method, "append_turn") == 0) handle_append_turn(ds, req, fd);
    else if (strcmp(req->method, "replace_history") == 0) handle_replace_history(ds, req, fd);
    else if (strcmp(req->method, "reset_session") == 0) handle_reset_session(ds, req, fd);
    else if (strcmp(req->method, "close_session") == 0) handle_close_session(ds, req, fd);
    else if (strcmp(req->method, "health") == 0) handle_health(ds, req, fd);
    else if (strcmp(req->method, "stats") == 0) handle_stats(ds, req, fd);
    else send_error(fd, req->id, -32601, "method not found");
}

/* ============================================================================
 * Server loop
 * ============================================================================ */

#define READ_BUF_SIZE 65536

static void signal_handler(int sig) {
    (void)sig;
    g_stop = 1;
}

/* Set read/write timeouts on a connected client socket. */
static void set_socket_timeouts(int fd, double read_sec, double write_sec) {
    struct timeval tv;
    if (read_sec > 0) {
        long sec = (long)read_sec;
        long usec = (long)((read_sec - sec) * 1e6);
        tv.tv_sec = sec;
        tv.tv_usec = usec;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }
    if (write_sec > 0) {
        long sec = (long)write_sec;
        long usec = (long)((write_sec - sec) * 1e6);
        tv.tv_sec = sec;
        tv.tv_usec = usec;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }
}

static int read_line(int fd, buf_t *b, size_t max_size) {
    buf_free(b);
    char tmp[4096];
    for (;;) {
        ssize_t n = read(fd, tmp, sizeof(tmp));
        if (n < 0) {
            if (errno == EINTR) {
                /* Honor shutdown signal even when a client keeps the
                 * connection open. Without this, SIGINT during a long-lived
                 * connection would wait up to SO_RCVTIMEO (-R) before
                 * returning. */
                if (g_stop) return -1;
                continue;
            }
            return -1;
        }
        if (n == 0) {
            return b->len > 0 ? 1 : 0;
        }
        if (b->len + (size_t)n > max_size) {
            return -2;
        }
        buf_append(b, tmp, (size_t)n);
        /* Check for newline. */
        for (size_t i = b->len - (size_t)n; i < b->len; i++) {
            if (b->ptr[i] == '\n') {
                return 1;
            }
        }
    }
}

static void handle_connection(daemon_state_t *ds, int fd, double idle_timeout, double request_timeout) {
    /* read_sec = request_timeout (per-request read), write_sec = idle_timeout (slow writer tolerance) */
    set_socket_timeouts(fd, request_timeout, idle_timeout);
    g_active_connections++;
    buf_t line = {0};
    for (;;) {
        if (g_stop) break;
        int rc = read_line(fd, &line, ds->max_request_size);
        if (rc == 0) break;
        if (rc < 0) {
            if (rc == -2) {
                send_error(fd, NULL, ERR_REQUEST_TOO_LARGE, "request too large");
            }
            break;
        }
        /* Trim trailing newline and any \r. */
        while (line.len > 0 && (line.ptr[line.len - 1] == '\n' || line.ptr[line.len - 1] == '\r')) {
            line.ptr[--line.len] = '\0';
        }
        if (line.len == 0) {
            buf_free(&line);
            line.ptr = NULL;
            continue;
        }

        request_t req = {0};
        if (!parse_request(line.ptr, &req)) {
            send_error(fd, NULL, -32600, "invalid request");
            buf_free(&line);
            line.ptr = NULL;
            continue;
        }
        if (!req.id) req.id = strdup("null");
        dispatch_request(ds, &req, fd);
        request_free(&req);
        buf_free(&line);
        line.ptr = NULL;
    }
    buf_free(&line);
    g_active_connections--;
}

static int create_unix_socket(const char *path) {
    unlink(path);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(addr.sun_path)) {
        close(fd);
        errno = ENAMETOOLONG;
        return -1;
    }
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    mode_t old_umask = umask(0077);
    int rc = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
    umask(old_umask);
    if (rc < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 16) < 0) {
        close(fd);
        return -1;
    }
    chmod(path, 0600);
    return fd;
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s -m <model.gguf> [options]\n"
        "Options:\n"
        "  -m PATH   Path to Qwen3 GGUF model\n"
        "  -s PATH   Unix Domain Socket path (default: /tmp/qwen3-engine.sock)\n"
        "  -c N      Context length / KV cache size (default: 4096)\n"
        "  -r BYTES  Max NDJSON request size (default: 1048576)\n"
        "  -t SEC    Single generate timeout (default: 600)\n"
        "  -T SEC    Connection idle timeout (default: 300)\n"
        "  -R SEC    Per-request read timeout (default: 60)\n"
        "  -k TYPE   KV-cache provider: local, service, fallback, none (default: none)\n"
        "  -K PATH   KV-cache service socket path (default: /tmp/ds3-kv-cache.sock)\n"
        "  -i ID     Daemon ID reported to the KV-cache service (default: pid)\n"
        "  -q        Quiet mode\n",
        argv0);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *socket_path = "/tmp/qwen3-engine.sock";
    const char *kv_provider_type = "none";
    const char *kv_service_path = "/tmp/ds3-kv-cache.sock";
    const char *kv_daemon_id = NULL;
    int n_ctx = 4096;
    size_t max_request_size = 1024 * 1024;
    double generate_timeout = 600.0;
    double idle_timeout = 300.0;
    double request_timeout = 60.0;
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            long v = atol(argv[++i]);
            if (v > 0) max_request_size = (size_t)v;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            generate_timeout = atof(argv[++i]);
            if (generate_timeout <= 0) generate_timeout = 600.0;
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            idle_timeout = atof(argv[++i]);
            if (idle_timeout <= 0) idle_timeout = 300.0;
        } else if (strcmp(argv[i], "-R") == 0 && i + 1 < argc) {
            request_timeout = atof(argv[++i]);
            if (request_timeout <= 0) request_timeout = 60.0;
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            kv_provider_type = argv[++i];
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            kv_service_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            kv_daemon_id = argv[++i];
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            quiet = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (quiet) ds3_set_quiet(1);

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    daemon_state_t ds = {0};
    ds.n_ctx = n_ctx;
    ds.max_request_size = max_request_size;
    ds.generate_timeout = generate_timeout;
    ds.started_at = time(NULL);

    if (strcmp(kv_provider_type, "local") == 0) {
        ds.kv_provider = ds3_kv_cache_provider_open(&ds3_kv_local_provider, NULL);
    } else if (strcmp(kv_provider_type, "service") == 0) {
        char service_config[320];
        if (kv_daemon_id && kv_daemon_id[0]) {
            snprintf(service_config, sizeof(service_config), "%s|%s", kv_service_path, kv_daemon_id);
        } else {
            snprintf(service_config, sizeof(service_config), "%s", kv_service_path);
        }
        ds.kv_provider = ds3_kv_cache_provider_open(&ds3_kv_service_provider, service_config);
    } else if (strcmp(kv_provider_type, "fallback") == 0) {
        ds.kv_provider = ds3_kv_cache_provider_open(&ds3_kv_fallback_provider, NULL);
    } else if (strcmp(kv_provider_type, "none") != 0) {
        fprintf(stderr, "Unknown KV provider type: %s\n", kv_provider_type);
        print_usage(argv[0]);
        return 1;
    }

    ds.engine = ds3_engine_open(model_path, n_ctx);
    if (!ds.engine) {
        fprintf(stderr, "Failed to load engine from %s\n", model_path);
        ds3_kv_cache_provider_close(ds.kv_provider);
        return 1;
    }
    if (ds.kv_provider) {
        ds3_engine_set_kv_provider(ds.engine, ds.kv_provider);
    }

    int listen_fd = create_unix_socket(socket_path);
    if (listen_fd < 0) {
        fprintf(stderr, "Failed to bind socket %s: %s\n", socket_path, strerror(errno));
        ds3_engine_close(ds.engine);
        return 1;
    }

    if (!quiet) {
        ds3_print_model_info(ds.engine);
        fprintf(stderr, "Listening on %s\n", socket_path);
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    while (!g_stop) {
        struct pollfd pfd = { .fd = listen_fd, .events = POLLIN };
        int rc = poll(&pfd, 1, 1000); /* 1 second timeout so g_stop is checked */
        if (rc < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "poll failed: %s\n", strerror(errno));
            break;
        }
        if (rc == 0) continue;

        struct sockaddr_un client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int fd = accept(listen_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "accept failed: %s\n", strerror(errno));
            break;
        }
        handle_connection(&ds, fd, idle_timeout, request_timeout);
        close(fd);
    }

    /* Graceful shutdown: stop accepting new connections and wait for in-flight
     * work to finish (up to the generate timeout, then force cleanup). */
    close(listen_fd);
    unlink(socket_path);

    if (!quiet) {
        fprintf(stderr, "Shutting down, waiting for active connections/generations...\n");
    }
    time_t shutdown_deadline = time(NULL) + (time_t)(generate_timeout > 0 ? generate_timeout : 60.0) + 5;
    while ((g_active_connections > 0 || g_active_generate > 0) && time(NULL) < shutdown_deadline) {
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    if (!quiet && (g_active_connections > 0 || g_active_generate > 0)) {
        fprintf(stderr, "Warning: forced shutdown with %d active connection(s) and %d active generation(s)\n",
                (int)g_active_connections, (int)g_active_generate);
    }
    for (int i = 0; i < ds.n_sessions; i++) {
        if (ds.kv_provider && ds.sessions[i]->provider_session_created) {
            ds3_kv_cache_close_session(ds.kv_provider, ds.sessions[i]->id);
        }
        session_free(ds.sessions[i]);
    }
    free(ds.sessions);
    ds3_engine_close(ds.engine);
    ds3_kv_cache_provider_close(ds.kv_provider);
    return 0;
}
