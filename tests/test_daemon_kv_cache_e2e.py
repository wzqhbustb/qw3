#!/usr/bin/env python3
"""End-to-end test: three-turn conversation through qwen3-engine-daemon
with the service KV-cache provider, verifying prefix-cache hits.

Requires:
  - ds3-kv-cache-svc running on /tmp/ds3-kv-cache.sock
  - qwen3-engine-daemon running on /tmp/qwen3-engine-e2e.sock
"""

import json
import socket
import struct
import sys
import time

DAEMON_SOCK = "/tmp/qwen3-engine-e2e.sock"


def rpc_call(method: str, params: dict, req_id: str = "1", timeout: float = 600.0) -> dict:
    # The daemon's JSON parser does not tolerate whitespace between tokens.
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params,
    }, separators=(',', ':')).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(DAEMON_SOCK)
        s.sendall(payload)
        resp_line = b""
        while not resp_line.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            resp_line += chunk
        # The engine may emit multi-byte UTF-8 cut mid-character; replace errors.
        return json.loads(resp_line.decode("utf-8", errors="replace"))


def main():
    session_id = "e2e-three-turns"
    system_prompt = "You are a helpful assistant. Answer in one word."

    print("create_session...")
    r1 = rpc_call("create_session", {
        "session_id": session_id,
        "system_prompt": system_prompt,
        "temperature": 0.0,
        "max_tokens": 2,
    }, req_id="1")
    print(f"create_session: {r1}")

    # Turn 1: establish a prompt of ~35 tokens.
    print("generate turn1...")
    t0 = time.time()
    g1 = rpc_call("generate", {
        "session_id": session_id,
        "user_prompt": "What is the capital of France?",
    }, req_id="2")
    dt1 = time.time() - t0
    r1 = g1.get("result", {})
    print(f"turn1 ({dt1:.2f}s): stats={r1.get('tokens_prompt')}/{r1.get('tokens_generated')}")

    # Turn 2: same prefix plus a follow-up. Should hit most of turn1's prompt.
    print("generate turn2...")
    t0 = time.time()
    g2 = rpc_call("generate", {
        "session_id": session_id,
        "user_prompt": "What is the capital of France? And Germany?",
    }, req_id="3")
    dt2 = time.time() - t0
    r2 = g2.get("result", {})
    print(f"turn2 ({dt2:.2f}s): stats={r2.get('tokens_prompt')}/{r2.get('tokens_generated')}")

    # Turn 3: same prefix plus another follow-up. Should hit even more.
    print("generate turn3...")
    t0 = time.time()
    g3 = rpc_call("generate", {
        "session_id": session_id,
        "user_prompt": "What is the capital of France? And Germany? And Italy?",
    }, req_id="4")
    dt3 = time.time() - t0
    r3 = g3.get("result", {})
    print(f"turn3 ({dt3:.2f}s): stats={r3.get('tokens_prompt')}/{r3.get('tokens_generated')}")

    ok = True
    # The prompts get longer each turn, so the key signal is latency being
    # much lower than a full cold prefill. turn1 >0.5s is the cold path;
    # subsequent turns should stay well under that even as the prompt grows.
    if dt2 >= dt1:
        print("WARNING: turn2 not faster than turn1")
        ok = False
    if dt3 >= dt1:
        print("WARNING: turn3 not faster than turn1")
        ok = False

    print(f"\nTiming summary: turn1={dt1:.2f}s turn2={dt2:.2f}s turn3={dt3:.2f}s")
    print("Check daemon logs for '[engine] KV cache hit: X / Y tokens' messages.")
    if ok:
        print("OK: prefix cache is reducing latency on subsequent turns")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
