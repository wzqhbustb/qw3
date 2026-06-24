/*
 * tests/test_kv_cache_service.c
 *
 * Integration test for the C service provider against ds3-kv-cache-svc.
 * Requires the Rust service to be running on /tmp/ds3-kv-cache.sock
 * (or the path passed via DS3_KVC_SOCKET env var).
 *
 * The test simulates a three-turn conversation and verifies that each
 * subsequent turn reuses the cached prefix from the previous turn.
 */

#include "ds3_kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#define SESSION_ID   "test-three-turns"
#define BLOCK_SIZE   16
#define PER_TOKEN_KV 98304
#define N_LAYER      48
#define N_KV_HEAD    4
#define HEAD_DIM     128

static int failures = 0;

static void check(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

/* Fill kv_buffer with a deterministic pattern for tokens [0, n_tokens). */
static void fill_kv(void *kv_buffer, int n_tokens)
{
    uint8_t *p = (uint8_t *)kv_buffer;
    size_t bytes = (size_t)n_tokens * PER_TOKEN_KV;
    for (size_t i = 0; i < bytes; i++) {
        p[i] = (uint8_t)((i * 7 + 13) & 0xFF);
    }
}

/* Verify that the first n_tokens of kv_buffer match the fill pattern. */
static int verify_kv(const void *kv_buffer, int n_tokens)
{
    const uint8_t *p = (const uint8_t *)kv_buffer;
    size_t bytes = (size_t)n_tokens * PER_TOKEN_KV;
    int mismatches = 0;
    for (size_t i = 0; i < bytes; i++) {
        uint8_t expected = (uint8_t)((i * 7 + 13) & 0xFF);
        if (p[i] != expected) {
            if (mismatches < 5) {
                fprintf(stderr, "  byte %zu: got %02x expected %02x\n", i, p[i], expected);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "  total mismatches: %d / %zu\n", mismatches, bytes);
        return -1;
    }
    return 0;
}

int main(void)
{
    const char *socket_path = getenv("DS3_KVC_SOCKET");
    if (!socket_path || !socket_path[0]) socket_path = "/tmp/ds3-kv-cache.sock";

    ds3_kv_cache_provider_t *provider = ds3_kv_cache_provider_open(
        &ds3_kv_service_provider, socket_path);
    if (!provider) {
        fprintf(stderr, "Failed to open service provider\n");
        return 1;
    }

    if (ds3_kv_cache_create_session(provider, SESSION_ID) != 0) {
        fprintf(stderr, "Failed to create session\n");
        ds3_kv_cache_provider_close(provider);
        return 1;
    }

    /* Turn 1: 24 tokens. */
    int turn1[24];
    for (int i = 0; i < 24; i++) turn1[i] = 100 + i;

    int cached = -1;
    int rc = ds3_kv_cache_lookup(provider, SESSION_ID, turn1, 24, &cached);
    check(rc == 0, "turn1 lookup rc");
    check(cached == 0, "turn1 lookup miss");

    void *kv1 = calloc(1, (size_t)24 * PER_TOKEN_KV);
    fill_kv(kv1, 24);
    rc = ds3_kv_cache_write(provider, SESSION_ID, turn1, 24, kv1, 24 * PER_TOKEN_KV);
    check(rc == 0, "turn1 write");
    free(kv1);

    /* Turn 2: previous 24 tokens + 8 new tokens. */
    int turn2[32];
    memcpy(turn2, turn1, sizeof(turn1));
    for (int i = 24; i < 32; i++) turn2[i] = 200 + (i - 24);

    cached = -1;
    rc = ds3_kv_cache_lookup(provider, SESSION_ID, turn2, 32, &cached);
    check(rc == 0, "turn2 lookup rc");
    check(cached == 24, "turn2 lookup hits 24 tokens");

    void *kv2 = calloc(1, (size_t)32 * PER_TOKEN_KV);
    fill_kv(kv2, 32);
    rc = ds3_kv_cache_write(provider, SESSION_ID, turn2, 32, kv2, 32 * PER_TOKEN_KV);
    check(rc == 0, "turn2 write");
    free(kv2);

    /* Turn 3: previous 32 tokens + 8 new tokens. */
    int turn3[40];
    memcpy(turn3, turn2, sizeof(turn2));
    for (int i = 32; i < 40; i++) turn3[i] = 300 + (i - 32);

    cached = -1;
    rc = ds3_kv_cache_lookup(provider, SESSION_ID, turn3, 40, &cached);
    check(rc == 0, "turn3 lookup rc");
    check(cached == 32, "turn3 lookup hits 32 tokens");

    /* Read back the cached 32-token prefix and verify its contents. */
    void *kv_read = calloc(1, (size_t)32 * PER_TOKEN_KV);
    rc = ds3_kv_cache_read(provider, SESSION_ID, 0, 32, kv_read);
    check(rc == 0, "turn3 read rc");
    check(verify_kv(kv_read, 32) == 0, "turn3 read data matches");
    free(kv_read);

    ds3_kv_cache_close_session(provider, SESSION_ID);
    ds3_kv_cache_provider_close(provider);

    if (failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("OK: three-turn prefix cache test passed\n");
    return 0;
}
