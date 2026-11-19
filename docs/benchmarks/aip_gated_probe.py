"""Gated tiny-2L bench: fire only when the GPU has been idle, to separate
co-tenancy contention from intrinsic variance (rec-ikf).

Each iteration waits up to QUIET_WAIT for gpu_busy_percent < 5 for 3
consecutive 0.3 s samples, records the pre-run busy/sclk stats, then runs
the un-instrumented bench once. Runs until MAX_RUNS gated samples or
MAX_SECS elapsed. Appends to /tmp/aip_var_gated.jsonl.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

DEV = Path("/sys/class/drm/card1/device")
PY = str(Path.home() / "rec-ve/.venv/bin/python")
CKPT = "output/syn-tiny"
ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    "LD_LIBRARY_PATH": "/nix/store/0iv8glcslgfcgn371lbjr5jjw5a6cqir-gcc-15.3.0-lib/lib:"
                       "/nix/store/13id30w3rvgj24nnz34f7qrncz48zd7l-zstd-1.5.7/lib",
}
OUT = Path("/tmp/aip_var_gated.jsonl")
MAX_RUNS, MAX_SECS, QUIET_WAIT = 8, 900, 120


def busy():
    return int((DEV / "gpu_busy_percent").read_text().strip())


def sclk_active():
    for line in (DEV / "pp_dpm_sclk").read_text().splitlines():
        if "*" in line:
            return "".join(ch for ch in line if ch.isdigit())
    return "?"


def wait_quiet():
    """Wait for 3 consecutive busy<5 samples. Returns (waited_secs, ok)."""
    t0 = time.time()
    streak = 0
    while time.time() - t0 < QUIET_WAIT:
        if busy() < 5:
            streak += 1
            if streak >= 3:
                return time.time() - t0, True
        else:
            streak = 0
        time.sleep(0.3)
    return time.time() - t0, False


def bench():
    r = subprocess.run([PY, "tests/torch_rocm_bench.py", "--ckpt", CKPT, "--device", "cuda"],
                       capture_output=True, text=True,
                       env={"PATH": "/run/current-system-sw/bin:/run/current-system/sw/bin:/home/dusk/.nix-profile/bin", **ENV},
                       cwd=str(Path.home() / "rec-aip-prep/engine"), timeout=600)
    return json.loads(r.stdout.strip())


def main():
    n = 0
    t_start = time.time()
    while n < MAX_RUNS and time.time() - t_start < MAX_SECS:
        waited, ok = wait_quiet()
        pre_busy = [busy() for _ in range(3)]
        if not ok or max(pre_busy) >= 5:
            print(f"run {n}: skipped (never quiet, waited {waited:.0f}s)", flush=True)
            continue
        d = bench()
        d["tag"] = f"gated_{n}"
        d["quiet_wait_s"] = round(waited, 1)
        d["pre_busy_max"] = max(pre_busy)
        d["sclk"] = sclk_active()
        with OUT.open("a") as f:
            f.write(json.dumps(d) + "\n")
        print(f"run {n}: pf {d['prefill512_tok_s']} dc {d['decode32_tok_s']} "
              f"(waited {waited:.0f}s, sclk {d['sclk']})", flush=True)
        n += 1
        time.sleep(2)
    print("GATED_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
