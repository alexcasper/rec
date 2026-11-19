"""Variance probe for the tiny-2L INT4 bench swing on dusk (rec-ebj follow-up).

Phases: cold tiny bench x5 -> 20 s dense-matmul clock-warm load -> warm tiny
bench x5 -> 30 s idle -> tiny bench x3. A sampler thread logs
gpu_busy_percent + active sclk/mclk from amdgpu sysfs at 0.3 s throughout.
Appends one JSON line per bench run to /tmp/aip_var_results.jsonl and the
raw bench stdout to /tmp/aip_var_results.txt.
"""
import json
import subprocess
import sys
import threading
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
RES = Path("/tmp/aip_var_results.jsonl")
TXT = Path("/tmp/aip_var_results.txt")

stop = threading.Event()


def sampler():
    while not stop.is_set():
        try:
            busy = (DEV / "gpu_busy_percent").read_text().strip()
            sclk = mclk = "?"
            for line in (DEV / "pp_dpm_sclk").read_text().splitlines():
                if "*" in line:
                    sclk = "".join(ch for ch in line if ch.isdigit())
            for line in (DEV / "pp_dpm_mclk").read_text().splitlines():
                if "*" in line:
                    mclk = "".join(ch for ch in line if ch.isdigit())
            RES.parent and None
            with open("/tmp/aip_var_samples.log", "a") as f:
                f.write(f"{time.time():.2f} busy={busy} sclk={sclk} mclk={mclk}\n")
        except Exception as e:
            with open("/tmp/aip_var_samples.log", "a") as f:
                f.write(f"{time.time():.2f} err {e}\n")
        stop.wait(0.3)


def bench(tag):
    r = subprocess.run([PY, "tests/torch_rocm_bench.py", "--ckpt", CKPT, "--device", "cuda"],
                       capture_output=True, text=True,
                       env={"PATH": "/run/current-system/sw/bin:/home/dusk/.nix-profile/bin", **ENV},
                       cwd=str(Path.home() / "rec-aip-prep/engine"), timeout=600)
    line = r.stdout.strip()
    with TXT.open("a") as f:
        f.write(f"--- {tag} ---\n{line}\n{r.stderr[-500:] if r.returncode else ''}\n")
    try:
        d = json.loads(line)
        d["tag"] = tag
        with RES.open("a") as f:
            f.write(json.dumps(d) + "\n")
        print(f"{tag}: prefill {d['prefill512_tok_s']} decode {d['decode32_tok_s']}", flush=True)
    except Exception:
        print(f"{tag}: FAILED {line[:200]}", flush=True)


def warm_load(seconds=20):
    code = (
        "import torch, time\n"
        "a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)\n"
        "t0 = time.time(); n = 0\n"
        "while time.time() - t0 < %d:\n"
        "    a @ a; n += 1\n"
        "torch.cuda.synchronize()\n"
        "print('warmload matmuls:', n)\n" % seconds
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       env={"PATH": "/run/current-system/sw/bin:/home/dusk/.nix-profile/bin", **ENV})
    print(r.stdout.strip() or r.stderr[-300:], flush=True)


def main():
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    for i in range(5):
        bench(f"cold_{i}")
    warm_load(20)
    for i in range(5):
        bench(f"warm_{i}")
    time.sleep(30)
    for i in range(3):
        bench(f"recool_{i}")
    stop.set()
    th.join(timeout=2)
    # clock level snapshot for the record
    print("sclk table:", (DEV / "pp_dpm_sclk").read_text().strip().replace("\n", " | "), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
