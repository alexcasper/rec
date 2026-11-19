"""One gated mid-8L profile run with full stdout redirected to a file
(the driver's 4 KB stdout capture truncates the profile JSON head).
"""
import subprocess
import sys
import time
from pathlib import Path

DEV = Path("/sys/class/drm/card1/device")
PY = str(Path.home() / "rec-ve/.venv/bin/python")
ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    "LD_LIBRARY_PATH": "/nix/store/0iv8glcslgfcgn371lbjr5jjw5a6cqir-gcc-15.3.0-lib/lib:"
                       "/nix/store/13id30w3rvgj24nnz34f7qrncz48zd7l-zstd-1.5.7/lib",
}


def busy():
    return int((DEV / "gpu_busy_percent").read_text().strip())


t0 = time.time()
streak = 0
while time.time() - t0 < 300:
    if busy() < 5:
        streak += 1
        if streak >= 3:
            break
    else:
        streak = 0
    time.sleep(0.3)
pre = max(busy() for _ in range(3))
print(f"waited {time.time()-t0:.0f}s pre_busy_max={pre}", flush=True)
if pre >= 5:
    print("ABORT: not quiet", flush=True)
    sys.exit(1)
r = subprocess.run([PY, "tests/aip_prep_profile.py", "--ckpt", "output/syn-mid",
                    "--device", "cuda", "--tag", "mid-8L-int4-gated-C"],
                   capture_output=True, text=True,
                   env={"PATH": "/run/current-system/sw/bin:/home/dusk/.nix-profile/bin", **ENV},
                   cwd=str(Path.home() / "rec-aip-prep/engine"), timeout=900)
out = Path.home() / "rec-aip-prep/aip_mid_gated_C.json"
out.write_text(r.stdout.strip())
print(f"rc={r.returncode} wrote {out} ({len(r.stdout)} bytes)", flush=True)
print("busy after:", busy(), flush=True)
