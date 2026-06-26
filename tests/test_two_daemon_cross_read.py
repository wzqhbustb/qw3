#!/usr/bin/env python3
"""Two-daemon cross-read test for the tiered KV cache service.

Starts one ds3-kv-cache-svc and two qwen3-engine-daemons.  Daemon1 creates
two distinct sessions with the same prompt, promoting the prefix to Global.
Daemon2 then creates a third session and should hit the cached prefix via a
read-only mapping of daemon1's SHM arena.
"""

import os
import subprocess
import sys
import time
import atexit
import json
import socket

KVCACHE_DIR = "/Users/wangyang/kv_cache"
SVC_BIN = os.path.join(KVCACHE_DIR, "target", "debug", "ds3-kv-cache-svc")
DAEMON_BIN = "/Users/wangyang/metal_demo/qwen3-engine-daemon"
MODEL_PATH = "/Users/wangyang/Qwen3-30B-A3B-Q4_K_M.gguf"

SVC_SOCK = "/tmp/ds3-kv-cache-two-daemon.sock"
PERSIST_DIR = "/tmp/ds3-kv-cache-two-daemon"
DAEMON1_SOCK = "/tmp/qwen3-engine-e2e-d1.sock"
DAEMON2_SOCK = "/tmp/qwen3-engine-e2e-d2.sock"


def wait_for_socket(path: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    return False


def rpc_call(sock_path: str, method: str, params: dict, req_id: str = "1", timeout: float = 300.0) -> dict:
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params,
    }, separators=(',', ':')).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall(payload)
        resp = b""
        while not resp.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
        return json.loads(resp.decode("utf-8", errors="replace"))


def start_service() -> subprocess.Popen:
    subprocess.run(["cargo", "build", "--quiet"], cwd=KVCACHE_DIR, check=True)
    for p in [SVC_SOCK, PERSIST_DIR]:
        try:
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p)
            else:
                os.remove(p)
        except FileNotFoundError:
            pass

    proc = subprocess.Popen(
        [
            SVC_BIN,
            "--tiered",
            "-s", SVC_SOCK,
            "-p", PERSIST_DIR,
            "-m", "134217728",
            "--max-daemons", "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=KVCACHE_DIR,
    )
    if not wait_for_socket(SVC_SOCK):
        print("Service failed to start", file=sys.stderr)
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        print(err.decode(), file=sys.stderr)
        sys.exit(1)
    return proc


def start_daemon(daemon_id: str, daemon_sock: str) -> subprocess.Popen:
    try:
        os.remove(daemon_sock)
    except FileNotFoundError:
        pass
    proc = subprocess.Popen(
        [
            DAEMON_BIN,
            "-m", MODEL_PATH,
            "-s", daemon_sock,
            "-c", "2048",
            "-k", "service",
            "-K", SVC_SOCK,
            "-i", daemon_id,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not wait_for_socket(daemon_sock):
        print(f"Daemon {daemon_id} failed to start", file=sys.stderr)
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        print(err.decode(), file=sys.stderr)
        sys.exit(1)
    return proc


def generate(daemon_sock: str, session_id: str, user_prompt: str, req_id: str) -> tuple:
    r = rpc_call(daemon_sock, "create_session", {
        "session_id": session_id,
        "system_prompt": "You are a helpful assistant. Answer in one word.",
        "temperature": 0.0,
        "max_tokens": 4,
    }, req_id=req_id)
    if "error" in r:
        raise RuntimeError(f"create_session failed: {r}")

    t0 = time.time()
    g = rpc_call(daemon_sock, "generate", {
        "session_id": session_id,
        "user_prompt": user_prompt,
    }, req_id=str(int(req_id) + 1))
    dt = time.time() - t0
    result = g.get("result", {})
    return dt, result


def main() -> int:
    svc = start_service()
    d1 = start_daemon("daemon1", DAEMON1_SOCK)
    d2 = start_daemon("daemon2", DAEMON2_SOCK)

    def cleanup():
        for p in [d2, d1, svc]:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for p in [DAEMON2_SOCK, DAEMON1_SOCK, SVC_SOCK]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        try:
            import shutil
            shutil.rmtree(PERSIST_DIR)
        except FileNotFoundError:
            pass

    atexit.register(cleanup)

    prompt = "What is the capital of France?"

    # d1 s1: cold prefill, stores SessionLocal prefix.
    dt1, r1 = generate(DAEMON1_SOCK, "s1", prompt, "1")
    print(f"d1 s1 turn1 ({dt1:.2f}s): {r1.get('text')!r} prompt={r1.get('tokens_prompt')} gen={r1.get('tokens_generated')}")

    # d1 s2: same prompt from a different session -> promotes to Global.
    # This is still a miss because the SessionLocal copy is not visible to s2,
    # but the store path promotes the node.
    dt2, r2 = generate(DAEMON1_SOCK, "s2", prompt, "3")
    print(f"d1 s2 turn1 ({dt2:.2f}s): {r2.get('text')!r} prompt={r2.get('tokens_prompt')} gen={r2.get('tokens_generated')}")

    # d2 s3: the prefix is now Global and owned by daemon1.  This must hit.
    dt3, r3 = generate(DAEMON2_SOCK, "s3", prompt, "5")
    print(f"d2 s3 turn1 ({dt3:.2f}s): {r3.get('text')!r} prompt={r3.get('tokens_prompt')} gen={r3.get('tokens_generated')}")

    ok = True
    # Cold prefill latency varies with system load; the reliable signal is that
    # the cross-daemon read is fast (< 1.5s) and produces the same answer.
    if dt3 >= 1.5:
        print("FAIL: d2 s3 should have hit daemon1's Global prefix (< 1.5s)")
        ok = False
    if r3.get("text") != r1.get("text"):
        print("FAIL: d2 s3 produced different output")
        ok = False

    if ok:
        print("OK: daemon2 successfully read daemon1's shared Global prefix")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
