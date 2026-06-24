#!/usr/bin/env python3
"""Single generate through daemon to verify KV cache store/lookup."""

import json
import socket
import sys
import time

DAEMON_SOCK = "/tmp/qwen3-engine-e2e.sock"


def rpc_call(method: str, params: dict, req_id: str = "1", timeout: float = 600.0) -> dict:
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
        rb = b""
        while not rb.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            rb += chunk
        return json.loads(rb.decode("utf-8", errors="replace"))


def main():
    session_id = "single-gen-test"
    print("create_session...")
    r = rpc_call("create_session", {
        "session_id": session_id,
        "system_prompt": "You are a helpful assistant. Answer in one word.",
        "temperature": 0.0,
        "max_tokens": 4,
    }, req_id="1")
    print(r)

    print("generate turn1...")
    t0 = time.time()
    g1 = rpc_call("generate", {
        "session_id": session_id,
        "user_prompt": "What is the capital of France?",
    }, req_id="2")
    dt1 = time.time() - t0
    print(f"turn1 ({dt1:.2f}s): {g1}")

    print("generate turn2...")
    t0 = time.time()
    g2 = rpc_call("generate", {
        "session_id": session_id,
        "user_prompt": "What is the capital of France? And Germany?",
    }, req_id="3")
    dt2 = time.time() - t0
    print(f"turn2 ({dt2:.2f}s): {g2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
