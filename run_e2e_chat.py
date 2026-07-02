#!/usr/bin/env python3
"""Simple NDJSON client for qwen3-engine-daemon."""

import argparse
import json
import socket
import sys


def send(sock, obj):
    # The daemon's minimal JSON parser does not accept whitespace.
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())


def recv_line(sock, timeout):
    sock.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(8192)
        if not chunk:
            break
        buf += chunk
    if not buf:
        return None
    line, _, _ = buf.partition(b"\n")
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon-socket", default="/tmp/qwen3-engine-e2e.sock")
    parser.add_argument(
        "--model",
        default="/Users/wangyang/Qwen3-30B-A3B-Q4_K_M.gguf",
    )
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.daemon_socket)

    req_id = 0

    def rpc(method, params, timeout=120):
        nonlocal req_id
        req_id += 1
        send(sock, {"id": str(req_id), "method": method, "params": params})
        return recv_line(sock, timeout)

    print("[client] load_model ...")
    resp = rpc("load_model", {"model_path": args.model, "n_ctx": args.n_ctx}, timeout=600)
    print("[client] load_model =>", resp)
    if not resp or not resp.get("result", {}).get("loaded"):
        print("[client] model load failed", file=sys.stderr)
        return 1

    print("[client] create_session ...")
    resp = rpc(
        "create_session",
        {
            "session_id": "chat",
            "system_prompt": "You are a helpful assistant.",
            "max_tokens": args.max_tokens,
        },
    )
    print("[client] create_session =>", resp)

    turns = [
        "What is the capital of France?",
        "What is its population?",
        "Tell me a fun fact about it.",
    ]

    for i, user_prompt in enumerate(turns, 1):
        print(f"[client] --- turn {i}: {user_prompt} ---")
        resp = rpc(
            "generate",
            {
                "session_id": "chat",
                "user_prompt": user_prompt,
                "max_tokens": args.max_tokens,
            },
            timeout=600,
        )
        print("[client] generate =>", resp)
        if resp and "result" in resp:
            print(f"[client] response text: {resp['result'].get('text', '')[:200]}")

    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
