CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -fobjc-arc
LDLIBS ?= -framework Foundation -framework Metal

# Qwen3 core sources (no Metal)
DS3_SRCS := src/ds3_gguf.c src/ds3_weights.c src/ds3_tokenizer.c src/ds3_reference.c src/ds3_log.c
DS3_ENGINE_SRCS := src/ds3_engine.c
DS3_OBJS := $(DS3_SRCS:.c=.o) $(DS3_ENGINE_SRCS:.c=.o)

.PHONY: clean run run_matmul test_gguf

# ── Metal demos ──────────────────────────────────────────

add_vectors: main.m add_vectors.metal
	$(CC) $(CFLAGS) -o $@ main.m $(LDLIBS)

matmul_tiled: main_matmul.m matmul_tiled.metal
	$(CC) $(CFLAGS) -o $@ main_matmul.m $(LDLIBS)

run: add_vectors
	./add_vectors

run_matmul: matmul_tiled
	./matmul_tiled

# ── GGUF loader test ─────────────────────────────────────

test_gguf: tests/test_gguf.c $(DS3_SRCS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_gguf.c $(DS3_SRCS)

# ── Tokenizer test ───────────────────────────────────────

test_tokenizer: tests/test_tokenizer.c $(DS3_SRCS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_tokenizer.c $(DS3_SRCS)

# ── Reference implementation test ────────────────────────

test_reference: tests/test_reference.c src/ds3_reference.c
	$(CC) $(CFLAGS) -I. -o $@ tests/test_reference.c src/ds3_reference.c -lm

# ── Layer 0 end-to-end test (Q8_0 GGUF required) ─────────

test_layer0_e2e: tests/test_layer0_e2e.c src/ds3_reference.c src/ds3_gguf.c src/ds3_weights.c src/ds3_log.c
	$(CC) $(CFLAGS) -I. -o $@ tests/test_layer0_e2e.c src/ds3_reference.c src/ds3_gguf.c src/ds3_weights.c src/ds3_log.c -lm

test: test_gguf test_tokenizer test_reference test_metal test_matmul_quant_metal test_attention_metal test_moe_metal
	@echo "Usage: ./test_gguf <path/to/model.gguf>"
	@echo "Usage: ./test_tokenizer <path/to/model.gguf> [text]"
	@echo "Usage: ./test_reference"
	@echo "Usage: ./test_layer0_e2e <path/to/Q8_0.gguf>"
	@echo "Usage: ./test_metal"
	@echo "Usage: ./test_matmul_quant_metal"
	@echo "Usage: ./test_attention_metal"
	@echo "Usage: ./test_moe_metal"

# ── Metal kernel tests ───────────────────────────────────

test_metal: tests/test_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c
	$(CC) $(CFLAGS) -I. -o $@ tests/test_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c $(LDLIBS) -lm

# ── Quantized matmul Metal test ──────────────────────────

test_matmul_quant_metal: tests/test_matmul_quant_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c
	$(CC) $(CFLAGS) -I. -o $@ tests/test_matmul_quant_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c $(LDLIBS) -lm

# ── Attention Metal test ─────────────────────────────────

test_attention_metal: tests/test_attention_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c
	$(CC) $(CFLAGS) -I. -o $@ tests/test_attention_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c $(LDLIBS) -lm

# ── MoE FFN Metal test ───────────────────────────────────

test_moe_metal: tests/test_moe_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c
	$(CC) $(CFLAGS) -I. -o $@ tests/test_moe_metal.c src/ds3_metal.m src/ds3_reference.c src/ds3_log.c $(LDLIBS) -lm

# ── Qwen3 engine CLI ─────────────────────────────────────

qwen3-cli: tools/qwen3-cli.c src/ds3_metal.m $(DS3_SRCS) $(DS3_ENGINE_SRCS)
	$(CC) $(CFLAGS) -I. -o $@ tools/qwen3-cli.c src/ds3_metal.m $(DS3_SRCS) $(DS3_ENGINE_SRCS) $(LDLIBS) -lm

run-cli: qwen3-cli
	@echo "Usage: ./qwen3-cli -m <model.gguf> -p '<prompt>'"

# ── Qwen3 engine socket daemon ───────────────────────────

qwen3-engine-daemon: tools/qwen3-engine-daemon.c src/ds3_metal.m $(DS3_SRCS) $(DS3_ENGINE_SRCS)
	$(CC) $(CFLAGS) -I. -o $@ tools/qwen3-engine-daemon.c src/ds3_metal.m $(DS3_SRCS) $(DS3_ENGINE_SRCS) $(LDLIBS) -lm

run-daemon: qwen3-engine-daemon
	@echo "Usage: ./qwen3-engine-daemon -m <model.gguf> -s /tmp/qwen3-engine.sock"

# ── Clean ────────────────────────────────────────────────

clean:
	rm -f add_vectors matmul_tiled qwen3-cli qwen3-engine-daemon test_gguf test_tokenizer test_reference test_layer0_e2e test_metal test_matmul_quant_metal test_attention_metal test_moe_metal
	rm -f $(DS3_OBJS)
	rm -f layer0_c_*.bin
