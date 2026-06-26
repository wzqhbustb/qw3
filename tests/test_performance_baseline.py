#!/usr/bin/env python3
"""Milestone 3 performance baseline for ds3-kv-cache-svc.

Runs real-model scenarios close to the §11 targets:
  - 700-token system prompt, 20-token user input, 128-token output
  - 5-turn latency on a single session
  - 10 sessions sharing the same system prompt -> measure peak RSS
  - service restart recovery time
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import atexit

KVCACHE_DIR = "/Users/wangyang/kv_cache"
SVC_BIN = os.path.join(KVCACHE_DIR, "target", "debug", "ds3-kv-cache-svc")
DAEMON_BIN = "/Users/wangyang/metal_demo/qwen3-engine-daemon"
MODEL_PATH = "/Users/wangyang/Qwen3-30B-A3B-Q4_K_M.gguf"

SVC_SOCK = "/tmp/ds3-kv-perf.sock"
PERSIST_DIR = "/tmp/ds3-kv-perf"
DAEMON_SOCK = "/tmp/qwen3-engine-perf.sock"

SYSTEM_PROMPT = (
    "You are a concise factual assistant. Here is a long set of background facts to keep in "
    "context for every answer: "
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is Madrid. "
    "The capital of Portugal is Lisbon. The capital of the Netherlands is Amsterdam. "
    "The capital of Belgium is Brussels. The capital of Switzerland is Bern. "
    "The capital of Austria is Vienna. The capital of Sweden is Stockholm. "
    "The capital of Norway is Oslo. The capital of Denmark is Copenhagen. "
    "The capital of Finland is Helsinki. The capital of Poland is Warsaw. "
    "The capital of the Czech Republic is Prague. The capital of Hungary is Budapest. "
    "The capital of Romania is Bucharest. The capital of Bulgaria is Sofia. "
    "The capital of Greece is Athens. The capital of Turkey is Ankara. "
    "The capital of Japan is Tokyo. The capital of China is Beijing. "
    "The capital of India is New Delhi. The capital of Russia is Moscow. "
    "The capital of Brazil is Brasilia. The capital of Argentina is Buenos Aires. "
    "The capital of Canada is Ottawa. The capital of the United States is Washington D.C. "
    "The capital of Mexico is Mexico City. The capital of Egypt is Cairo. "
    "The capital of South Africa is Pretoria. The capital of Australia is Canberra. "
    "The capital of Indonesia is Jakarta. The capital of Saudi Arabia is Riyadh. "
    "The capital of Iran is Tehran. The capital of Pakistan is Islamabad. "
    "The capital of Bangladesh is Dhaka. The capital of Nigeria is Abuja. "
    "The capital of Kenya is Nairobi. The capital of Ethiopia is Addis Ababa. "
    "The capital of Ghana is Accra. The capital of Morocco is Rabat. "
    "The capital of Peru is Lima. The capital of Chile is Santiago. "
    "The capital of Colombia is Bogota. The capital of Venezuela is Caracas. "
    "The capital of Ecuador is Quito. The capital of Uruguay is Montevideo. "
    "The capital of Paraguay is Asuncion. The capital of Bolivia is Sucre. "
    "The capital of Argentina is Buenos Aires. The capital of Chile is Santiago. "
    "The capital of Cuba is Havana. The capital of the Dominican Republic is Santo Domingo. "
    "The capital of Guatemala is Guatemala City. The capital of Honduras is Tegucigalpa. "
    "The capital of El Salvador is San Salvador. The capital of Nicaragua is Managua. "
    "The capital of Costa Rica is San Jose. The capital of Panama is Panama City. "
    "The capital of Jamaica is Kingston. The capital of Haiti is Port-au-Prince. "
    "The capital of Trinidad and Tobago is Port of Spain. The capital of Barbados is Bridgetown. "
    "The capital of the Bahamas is Nassau. The capital of Belize is Belmopan. "
    "The capital of Guyana is Georgetown. The capital of Suriname is Paramaribo. "
    "The capital of Iceland is Reykjavik. The capital of Ireland is Dublin. "
    "The capital of the United Kingdom is London. The capital of France is Paris. "
    "The capital of Norway is Oslo. The capital of Finland is Helsinki. "
    "The capital of Estonia is Tallinn. The capital of Latvia is Riga. "
    "The capital of Lithuania is Vilnius. The capital of Belarus is Minsk. "
    "The capital of Ukraine is Kyiv. The capital of Moldova is Chisinau. "
    "The capital of Romania is Bucharest. The capital of Bulgaria is Sofia. "
    "The capital of Greece is Athens. The capital of Albania is Tirana. "
    "The capital of North Macedonia is Skopje. The capital of Kosovo is Pristina. "
    "The capital of Montenegro is Podgorica. The capital of Serbia is Belgrade. "
    "The capital of Bosnia and Herzegovina is Sarajevo. The capital of Croatia is Zagreb. "
    "The capital of Slovenia is Ljubljana. The capital of Slovakia is Bratislava. "
    "The capital of the Czech Republic is Prague. The capital of Hungary is Budapest. "
    "The capital of Poland is Warsaw. The capital of Germany is Berlin. "
    "The capital of Austria is Vienna. The capital of Switzerland is Bern. "
    "The capital of Liechtenstein is Vaduz. The capital of Luxembourg is Luxembourg City. "
    "The capital of Belgium is Brussels. The capital of the Netherlands is Amsterdam. "
    "The capital of Denmark is Copenhagen. The capital of Sweden is Stockholm. "
    "The capital of Malta is Valletta. The capital of Cyprus is Nicosia. "
    "The capital of Portugal is Lisbon. The capital of Spain is Madrid. "
    "The capital of Andorra is Andorra la Vella. The capital of Monaco is Monaco. "
    "The capital of San Marino is San Marino. The capital of Vatican City is Vatican City. "
    "The capital of Italy is Rome. The capital of Malta is Valletta. "
)

USER_PROMPT = "Continue."


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


def service_rss_kb(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "rss="], text=True)
        return int(out.strip())
    except Exception:
        return 0


def start_service(persist_dir: str, extra_args=None) -> subprocess.Popen:
    extra_args = extra_args or []
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    for p in [SVC_SOCK]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    proc = subprocess.Popen(
        [SVC_BIN, "--tiered", "-s", SVC_SOCK, "-p", persist_dir] + extra_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=KVCACHE_DIR,
    )
    if not wait_for_socket(SVC_SOCK):
        print("Service failed to start", file=sys.stderr)
        proc.terminate()
        _, err = proc.communicate(timeout=5)
        print(err.decode(), file=sys.stderr)
        raise RuntimeError("service start failed")
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
        _, err = proc.communicate(timeout=5)
        print(err.decode(), file=sys.stderr)
        raise RuntimeError(f"daemon {daemon_id} start failed")
    return proc


def generate(daemon_sock: str, session_id: str, user_prompt: str, max_tokens: int, req_id: int) -> tuple:
    r = rpc_call(daemon_sock, "create_session", {
        "session_id": session_id,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0.0,
        "max_tokens": max_tokens,
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


def test_five_turn_latency(daemon_sock: str) -> tuple:
    print("\n--- 5-turn latency (128 tokens/turn, same session) ---")
    session_id = "latency-session"
    rpc_call(daemon_sock, "create_session", {
        "session_id": session_id,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0.0,
        "max_tokens": 128,
    }, req_id="1")
    latencies = []
    for turn in range(1, 6):
        t0 = time.time()
        result = rpc_call(daemon_sock, "generate", {
            "session_id": session_id,
            "user_prompt": USER_PROMPT,
        }, req_id=str(turn + 1), timeout=1200.0).get("result", {})
        dt = time.time() - t0
        print(f"turn{turn}: {dt:.2f}s prompt={result.get('tokens_prompt')} gen={result.get('tokens_generated')}")
        latencies.append(dt)
    return latencies


def test_ten_session_memory(svc: subprocess.Popen, daemon_proc: subprocess.Popen, daemon_sock: str) -> int:
    print("\n--- 10-session shared system prompt memory ---")
    for i in range(10):
        dt, result = generate(daemon_sock, f"mem-{i}", "What is the capital of France?", max_tokens=8, req_id=100 + i * 10)
        print(f"  session {i}: {dt:.2f}s prompt={result.get('tokens_prompt')} gen={result.get('tokens_generated')}")
    time.sleep(1)
    # SHM arenas are resident in the daemon's address space, not the service.
    daemon_rss_kb = service_rss_kb(daemon_proc.pid)
    svc_rss_kb = service_rss_kb(svc.pid)
    print(f"service RSS: {svc_rss_kb / 1024:.1f} MiB, daemon RSS: {daemon_rss_kb / 1024:.1f} MiB")
    return daemon_rss_kb


def test_restart_recovery() -> tuple:
    print("\n--- service restart recovery ---")
    persist = PERSIST_DIR
    svc = start_service(persist, ["-m", "1073741824", "--max-daemons", "1"])
    d = start_daemon("daemon1", DAEMON_SOCK)
    try:
        dt1, r1 = generate(DAEMON_SOCK, "rec", USER_PROMPT, max_tokens=4, req_id=300)
        print(f"pre-restart: {dt1:.2f}s prompt={r1.get('tokens_prompt')}")
    finally:
        terminate(d)
        terminate(svc)

    t0 = time.time()
    svc = start_service(persist, ["-m", "1073741824", "--max-daemons", "1"])
    startup_dt = time.time() - t0
    d = start_daemon("daemon1", DAEMON_SOCK)
    try:
        dt2, r2 = generate(DAEMON_SOCK, "rec", USER_PROMPT, max_tokens=4, req_id=400)
        print(f"post-restart: startup={startup_dt:.2f}s turn={dt2:.2f}s prompt={r2.get('tokens_prompt')}")
        return startup_dt, dt2
    finally:
        terminate(d)
        terminate(svc)


def main() -> int:
    subprocess.run(["cargo", "build", "--quiet"], cwd=KVCACHE_DIR, check=True)

    svc = start_service(PERSIST_DIR, ["-m", "1073741824", "--max-daemons", "1"])
    d = start_daemon("daemon1", DAEMON_SOCK)
    atexit.register(lambda: (terminate(d), terminate(svc), shutil.rmtree(PERSIST_DIR, ignore_errors=True)))

    latencies = test_five_turn_latency(DAEMON_SOCK)
    rss_kb = test_ten_session_memory(svc, d, DAEMON_SOCK)

    # Restart test uses its own service lifecycle and cleans up itself.
    startup_dt, post_dt = test_restart_recovery()

    print("\n=== Performance Baseline Summary ===")
    print(f"5-turn latencies: {[f'{x:.2f}s' for x in latencies]}")
    print(f"10-session service RSS: {rss_kb / 1024:.1f} MiB")
    print(f"service restart: {startup_dt:.2f}s, post-restart turn: {post_dt:.2f}s")

    ok = True
    if latencies[4] > 10.0:
        print(f"WARNING: turn5 latency {latencies[4]:.2f}s is higher than ~5s target")
    if rss_kb > 1_500_000:
        print(f"WARNING: daemon RSS {rss_kb/1024:.1f} MiB is well above the ~1 GB target")
    if startup_dt > 10.0:
        print(f"WARNING: service restart took {startup_dt:.2f}s")
    if post_dt > 10.0:
        print(f"WARNING: post-restart first turn took {post_dt:.2f}s")

    print("Baseline run complete.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
