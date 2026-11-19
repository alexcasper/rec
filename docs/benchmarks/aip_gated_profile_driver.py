"""Gated mid-8L re-profile driver (rec-ikf follow-up): waits for an idle GPU
(gpu_busy_percent < 5 for 3 consecutive 0.3 s samples) before each invocation
so co-tenant queues cannot inflate stage timings, then runs the un-instrumented
bench and the stage profiler. Appends a JSON line per invocation to
/tmp/aip_var_gated_mid.jsonl with pre-run busy stats.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

DEV = Path("/sys/class/drm/card1/device")
PY = str(Path.home() / "rec-ve/.venv/bin/python")
CKPT = "output/syn-mid"
ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    "LD_LIBRARY_PATH": "/nix/store/0iv8glcslgfcgn371lbjr5jjw5a6cqir-gcc-15.3.0-lib/lib:"
                       "/nix/store/13id30w3rvgj24nnz34f7qrncz48zd7l-zstd-1.5.7/lib",
}
OUT = Path("/tmp/aip_var_gated_mid.jsonl")
QUIET_WAIT = 300


def busy():
    return int((DEV / "gpu_busy_percent").read_text().strip())


def wait_quiet():
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


def run(tag, args, timeout):
    waited, ok = wait_quiet()
    pre = max(busy() for _ in range(3))
    if not ok or pre >= 5:
        print(f"{tag}: SKIPPED never-quiet (waited {waited:.0f}s)", flush=True)
        return None
    t0 = time.time()
    r = subprocess.run([PY] + args, capture_output=True, text=True,
                       env={"PATH": "/run/current-system/sw/bin:/home/dusk/.nix-profile/bin", **ENV},
                       cwd=str(Path.home() / "rec-aip-prep/engine"), timeout=timeout)
    wall = time.time() - t0
    post_busy = busy()
    rec = {"tag": tag, "quiet_wait_s": round(waited, 1), "pre_busy_max": pre,
           "wall_s": round(wall, 1), "post_busy": post_busy,
           "stdout": r.stdout.strip()[-4000:], "rc": r.returncode}
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{tag}: rc={r.returncode} wall={wall:.0f}s waited={waited:.0f}s "
          f"post_busy={post_busy}", flush=True)
    print((r.stdout.strip().splitlines() or [""])[-1][:400], flush=True)
    return rec


def main():
    specs = [
        ("bench_gated", ["tests/torch_rocm_bench.py", "--ckpt", CKPT, "--device", "cuda"], 600),
        ("profile_A", ["tests/aip_prep_profile.py", "--ckpt", CKPT, "--device", "cuda",
                       "--tag", "mid-8L-int4-gated-A"], 900),
        ("profile_B", ["tests/aip_prep_profile.py", "--ckpt", CKPT, "--device", "cuda",
                       "--tag", "mid-8L-int4-gated-B"], 900),
    ]
    for tag, args, tmo in specs:
        if run(tag, args, tmo) is None:
            print("ABORT: GPU never quiet enough for a trustworthy profile", flush=True)
            return 1
    print("GATED_PROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
