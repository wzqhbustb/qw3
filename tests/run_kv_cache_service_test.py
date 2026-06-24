#!/usr/bin/env python3
"""Run the C service-provider integration test against ds3-kv-cache-svc."""

import os
import subprocess
import sys
import time
import atexit

SOCKET_PATH = "/tmp/ds3-kv-cache-test.sock"
MMAP_PATH = "/tmp/ds3-kv-cache-test.bin"
KVCACHE_DIR = "/Users/wangyang/kv_cache"
SVC_BIN = os.path.join(KVCACHE_DIR, "target", "debug", "ds3-kv-cache-svc")
TEST_BIN = "./test_kv_cache_service"


def main():
    # Clean up any leftover state.
    for p in [SOCKET_PATH, MMAP_PATH]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    # Build the Rust service if needed.
    subprocess.run(["cargo", "build", "--quiet"], cwd=KVCACHE_DIR, check=True)

    # Start the service.
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    proc = subprocess.Popen(
        [
            SVC_BIN,
            "-s", SOCKET_PATH,
            "-p", MMAP_PATH,
            "-m", "67108864",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=KVCACHE_DIR,
    )

    def cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for p in [SOCKET_PATH, MMAP_PATH]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    atexit.register(cleanup)

    # Wait for the socket to appear.
    for _ in range(60):
        if os.path.exists(SOCKET_PATH):
            break
        time.sleep(0.1)
    else:
        print("Service failed to start", file=sys.stderr)
        stdout, stderr = proc.communicate(timeout=5)
        print(stderr.decode(), file=sys.stderr)
        return 1

    # Run the C test.
    test_env = os.environ.copy()
    test_env["DS3_KVC_SOCKET"] = SOCKET_PATH
    result = subprocess.run([TEST_BIN], env=test_env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
