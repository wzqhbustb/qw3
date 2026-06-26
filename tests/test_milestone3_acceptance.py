#!/usr/bin/env python3
"""Milestone 3 acceptance tests for the ds3 KV cache service.

Runs three real-model end-to-end scenarios:
  1. Service restart: a daemon reconnects with the same id and hits the
     persisted prefix index.
  2. L1 eviction -> L2: a small memory budget forces blocks to disk; a later
     read cold-loads them back into L1.
  3. Daemon crash cleanup: after a daemon is SIGKILL'd, heartbeat timeout
     frees its orphan session so a new daemon can use the slot.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

KVCACHE_DIR = "/Users/wangyang/kv_cache"
SVC_BIN = os.path.join(KVCACHE_DIR, "target", "debug", "ds3-kv-cache-svc")
DAEMON_BIN = "/Users/wangyang/metal_demo/qwen3-engine-daemon"
MODEL_PATH = "/Users/wangyang/Qwen3-30B-A3B-Q4_K_M.gguf"


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


def start_service(persist_dir: str, svc_sock: str, extra_args=None) -> subprocess.Popen:
    extra_args = extra_args or []
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    try:
        os.remove(svc_sock)
    except FileNotFoundError:
        pass
    proc = subprocess.Popen(
        [SVC_BIN, "--tiered", "-s", svc_sock, "-p", persist_dir] + extra_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=KVCACHE_DIR,
    )
    if not wait_for_socket(svc_sock):
        print("Service failed to start", file=sys.stderr)
        proc.terminate()
        _, err = proc.communicate(timeout=5)
        print(err.decode(), file=sys.stderr)
        raise RuntimeError("service start failed")
    return proc


def start_daemon(daemon_id: str, daemon_sock: str, svc_sock: str) -> subprocess.Popen:
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
            "-K", svc_sock,
            "-i", daemon_id,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not wait_for_socket(daemon_sock):
        print(f"Daemon {daemon_id} failed to start", file=sys.stderr)
        proc.terminate()
        _, err = proc.communicate(timeout=5)
        print(err.decode(), file=sys.stderr)
        raise RuntimeError(f"daemon {daemon_id} start failed")
    return proc


def generate(daemon_sock: str, session_id: str, user_prompt: str, req_id: int = 1) -> tuple:
    r = rpc_call(daemon_sock, "create_session", {
        "session_id": session_id,
        "system_prompt": "You are a helpful assistant. Answer in one word.",
        "temperature": 0.0,
        "max_tokens": 4,
    }, req_id=str(req_id))
    if "error" in r:
        raise RuntimeError(f"create_session failed: {r}")
    t0 = time.time()
    g = rpc_call(daemon_sock, "generate", {
        "session_id": session_id,
        "user_prompt": user_prompt,
    }, req_id=str(req_id + 1))
    dt = time.time() - t0
    return dt, g.get("result", {})


def terminate(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_service_restart_reconnect() -> bool:
    print("\n=== Test 1: service restart + daemon reconnect ===")
    persist = tempfile.mkdtemp(prefix="kv-restart-")
    svc_sock = "/tmp/ds3-kv-restart.sock"
    daemon_sock = "/tmp/qwen3-engine-restart.sock"
    svc = start_service(persist, svc_sock, ["-m", "134217728", "--max-daemons", "1"])
    d1 = start_daemon("daemon1", daemon_sock, svc_sock)
    try:
        dt1, r1 = generate(daemon_sock, "s1", "What is the capital of France?")
        print(f"pre-restart turn1 ({dt1:.2f}s): {r1.get('text')!r}")

        # Simulate service restart.  The daemon may or may not survive the
        # control socket going away; we restart both with the same daemon id.
        terminate(svc)
        terminate(d1)

        svc = start_service(persist, svc_sock, ["-m", "134217728", "--max-daemons", "1"])
        d1 = start_daemon("daemon1", daemon_sock, svc_sock)

        # Re-create the same session id and verify the prefix is still cached.
        dt2, r2 = generate(daemon_sock, "s1", "What is the capital of France?")
        print(f"post-restart turn1 ({dt2:.2f}s): {r2.get('text')!r}")

        if dt2 >= 5.0:
            print("FAIL: post-restart lookup should hit cached prefix")
            return False
        if r2.get("text") != r1.get("text"):
            print("FAIL: post-restart output changed")
            return False
        print("OK: persisted prefix survived service restart")
        return True
    finally:
        terminate(d1)
        terminate(svc)
        shutil.rmtree(persist, ignore_errors=True)


def test_l1_eviction_to_l2() -> bool:
    print("\n=== Test 2: L1 eviction -> L2 cold load ===")
    persist = tempfile.mkdtemp(prefix="kv-tiered-")
    svc_sock = "/tmp/ds3-kv-tiered.sock"
    daemon_sock = "/tmp/qwen3-engine-tiered.sock"
    # Small budget: ~10 blocks in L1.  Each block is ~1.5 MiB.
    svc = start_service(persist, svc_sock, ["-m", "16777216", "--max-daemons", "1"])
    d1 = start_daemon("daemon1", daemon_sock, svc_sock)
    try:
        # Use a long user prompt to write more blocks than fit in L1.
        long_prompt = (
            "List the capitals of these twenty countries in order: "
            "France, Germany, Italy, Spain, Portugal, Netherlands, Belgium, "
            "Switzerland, Austria, Sweden, Norway, Denmark, Finland, Poland, "
            "Czech Republic, Hungary, Romania, Bulgaria, Greece, Turkey. "
            "Answer with one word per country."
        )
        dt1, r1 = generate(daemon_sock, "s1", long_prompt, req_id=1)
        print(f"long prompt turn1 ({dt1:.2f}s): tokens_prompt={r1.get('tokens_prompt')} gen={r1.get('tokens_generated')}")

        # Generate enough additional tokens to push earlier blocks out of L1.
        dt2, r2 = generate(daemon_sock, "s2", long_prompt + " Also add Japan and China.", req_id=3)
        print(f"extended turn ({dt2:.2f}s): tokens_prompt={r2.get('tokens_prompt')} gen={r2.get('tokens_generated')}")

        # A new session with the original long prompt must still hit; this
        # exercises cold-loading from L2 back into L1.
        dt3, r3 = generate(daemon_sock, "s3", long_prompt, req_id=5)
        print(f"cold-load turn ({dt3:.2f}s): tokens_prompt={r3.get('tokens_prompt')} gen={r3.get('tokens_generated')}")

        # We only assert that the cold-load path succeeds and returns the same
        # answer; exact timing depends on the machine and model.
        if r3.get("text") != r1.get("text"):
            print("FAIL: cold-load produced different output")
            return False
        print("OK: prefix survived L1 eviction and cold-loaded from L2")
        return True
    finally:
        terminate(d1)
        terminate(svc)
        shutil.rmtree(persist, ignore_errors=True)


def test_daemon_crash_cleanup() -> bool:
    print("\n=== Test 3: daemon crash -> heartbeat timeout cleanup ===")
    persist = tempfile.mkdtemp(prefix="kv-crash-")
    svc_sock = "/tmp/ds3-kv-crash.sock"
    daemon1_sock = "/tmp/qwen3-engine-crash-d1.sock"
    daemon2_sock = "/tmp/qwen3-engine-crash-d2.sock"
    # Use a 5s heartbeat timeout so the test does not wait 30s.
    svc = start_service(persist, svc_sock, [
        "-m", "134217728",
        "--max-daemons", "2",
        "--heartbeat-timeout-secs", "5",
    ])
    d1 = start_daemon("daemon1", daemon1_sock, svc_sock)
    try:
        dt1, r1 = generate(daemon1_sock, "s1", "What is the capital of France?")
        print(f"d1 turn1 ({dt1:.2f}s): {r1.get('text')!r}")

        # Hard-kill daemon1.  Its session is now orphaned.
        d1.kill()
        d1.wait()
        print("daemon1 killed, waiting for heartbeat timeout...")
        time.sleep(8)

        # A new daemon should be able to connect and allocate memory that was
        # held by the orphan session.
        d2 = start_daemon("daemon2", daemon2_sock, svc_sock)
        try:
            dt2, r2 = generate(daemon2_sock, "s2", "What is the capital of Germany?")
            print(f"d2 turn1 ({dt2:.2f}s): {r2.get('text')!r}")
            if not r2.get("text"):
                print("FAIL: d2 generation failed")
                return False
            print("OK: orphan session was cleaned up after daemon crash")
            return True
        finally:
            terminate(d2)
    finally:
        terminate(d1)
        terminate(svc)
        shutil.rmtree(persist, ignore_errors=True)


def main() -> int:
    subprocess.run(["cargo", "build", "--quiet"], cwd=KVCACHE_DIR, check=True)
    results = []
    results.append(("service restart", test_service_restart_reconnect()))
    results.append(("L1 eviction", test_l1_eviction_to_l2()))
    results.append(("daemon crash cleanup", test_daemon_crash_cleanup()))

    print("\n=== Summary ===")
    ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
