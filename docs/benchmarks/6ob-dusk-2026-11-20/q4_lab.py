"""rec-6ob lab driver: byte-identity + effective bandwidth of q4 gemv load-path variants vs production.

V0 = production `flint_q4_gemv_host` from flint_mid.so (built alone from the untouched run-of-record
source); V1..V4 from q4_lab.so (see q4_lab.hip); V9 = streaming read of the same weight bytes (ceiling).
Inputs are run_q4_bench's (seed 42). Every variant's fp32 output must be byte-identical to V0's.
Timing: A9 method (warmup 10, 50 back-to-back launches between events), bandwidth formula of
run-definitions A9 (weights + scales + biases + x + fp32 out). Variants run in ABBA order per shape
(forward, reverse), each round busy-gated; the per-variant figure is the median over rounds.

    python q4_lab.py --prod flint_mid.so --lab q4_lab.so [--grids 0,16] [--rounds 2] [--out lab.jsonl]
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

SHAPES = [  # mid-8L decode projections + the lm_head (rows, K)
    ("expert_gate_up_512x2048", 512, 2048),
    ("expert_down_2048x512", 2048, 512),
    ("shared_gate_up_1024x2048", 1024, 2048),
    ("shared_down_2048x1024", 2048, 1024),
    ("kv_proj_512x2048", 512, 2048),
    ("q_proj_4096x2048", 4096, 2048),
    ("o_proj_2048x4096", 2048, 4096),
    ("dense_gate_up_2048x2048", 2048, 2048),
    ("lm_head_150144x2048", 150144, 2048),
]
VARIANTS = {0: "prod", 1: "v1_inline", 2: "v2_batched", 3: "v3_prefetch", 4: "v4_rows8", 9: "stream_ceiling",
            10: "gx_prod_op", 11: "gx_generic_batched", 12: "gx_batched_lds"}
H = "100.98.73.29"


def f32_to_bf16(a: np.ndarray) -> np.ndarray:  # RNE, as mid_mk_driver.f32_to_bf16
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def busy() -> int:
    for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            return int(open(p).read())
        except OSError:
            pass
    return 100


def gate(timeout=300.0) -> bool:
    t0, ok = time.time(), 0
    while time.time() - t0 < timeout:
        ok = ok + 1 if busy() < 5 else 0
        if ok >= 3:
            return True
        time.sleep(0.3)
    return False


def cotenants() -> str:
    st = []
    for name, port in (("flint", 8080), ("e4b", 8090), ("qwen", 8091)):
        try:
            out = subprocess.run(["curl", "-s", "-m", "1", f"http://{H}:{port}/slots"], capture_output=True,
                                 text=True, timeout=3).stdout
        except Exception:
            out = ""
        st.append(f"{name}:{('busy' if '\"is_processing\":true' in out else 'idle') if out else 'down'}")
    return ",".join(st)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", type=Path, required=True)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--grids", default="0,16")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--variants", default="0,1,2,3,4,9")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--out", type=Path, default=Path("lab.jsonl"))
    a = ap.parse_args()
    prod, lab = ctypes.CDLL(str(a.prod.resolve())), ctypes.CDLL(str(a.lab.resolve()))
    vp = ctypes.c_void_p
    prod.flint_q4_gemv_host.restype = ctypes.c_int
    prod.flint_q4_gemv_host.argtypes = [vp] * 6 + [ctypes.c_int] * 5 + [vp]
    lab.lab_run.restype = ctypes.c_int
    lab.lab_run.argtypes = [ctypes.c_int, vp, vp, vp, vp, vp] + [ctypes.c_int] * 5 + [vp]
    lab.lab_alloc.restype = vp
    lab.lab_alloc.argtypes = [ctypes.c_size_t]
    for fn in ("lab_h2d", "lab_d2h"):
        getattr(lab, fn).argtypes = [vp, vp, ctypes.c_size_t]
    lab.lab_free.argtypes = [vp]
    lab.lab_memset.argtypes = [vp, ctypes.c_int, ctypes.c_size_t]
    variants = [int(v) for v in a.variants.split(",")]
    shapes = [s for s in SHAPES if not a.shapes or s[0] in a.shapes.split(",")]
    log = open(a.out, "a")
    rng = np.random.default_rng(42)
    summary = []
    for label, n, k in shapes:
        nib = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
        s_f = ((rng.random((n, k // 64)) * 0.5 + 0.75) * (0.13 / np.sqrt(k))).astype(np.float32)
        b_f = (-7.5 * s_f * (1.0 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32)
        x_f = rng.standard_normal(k).astype(np.float32)
        s_bf, b_bf, x_bf = f32_to_bf16(s_f), f32_to_bf16(b_f), f32_to_bf16(x_f)
        packed = np.ascontiguousarray((nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8))
        del nib
        nbytes = packed.nbytes + s_bf.nbytes + b_bf.nbytes + x_bf.nbytes + n * 4
        bufs = {}
        for name, arr in (("w", packed), ("s", s_bf), ("b", b_bf), ("x", x_bf)):
            bufs[name] = lab.lab_alloc(arr.nbytes)
            assert bufs[name] and lab.lab_h2d(bufs[name], arr.ctypes.data, arr.nbytes) == 0
        bufs["o"] = lab.lab_alloc(n * 4)
        xf = np.ascontiguousarray((x_bf.astype(np.uint32) << 16).view(np.float32))  # exact bf16 -> fp32
        bufs["xf"] = lab.lab_alloc(xf.nbytes)
        assert lab.lab_h2d(bufs["xf"], xf.ctypes.data, xf.nbytes) == 0
        for grid in [int(g) for g in a.grids.split(",")]:
            ref = np.zeros(n, np.float32)
            ms0 = ctypes.c_float(0.0)
            assert prod.flint_q4_gemv_host(packed.ctypes.data, s_bf.ctypes.data, b_bf.ctypes.data, x_bf.ctypes.data,
                                           None, ref.ctypes.data, n, k, grid, 1, 1, ctypes.byref(ms0)) == 0
            times = {v: [] for v in variants}
            ident = {}
            for rnd in range(a.rounds):
                order = variants if rnd % 2 == 0 else variants[::-1]
                if not gate():
                    print(f"GATE TIMEOUT {label} grid={grid}", flush=True)
                    return 2
                pre = f"busy={busy()} {cotenants()}"
                for v in order:
                    out = np.zeros(n, np.float32)
                    ms = ctypes.c_float(0.0)
                    if v == 0:
                        rc = prod.flint_q4_gemv_host(packed.ctypes.data, s_bf.ctypes.data, b_bf.ctypes.data,
                                                     x_bf.ctypes.data, None, out.ctypes.data, n, k, grid, 10, 50,
                                                     ctypes.byref(ms))
                    else:
                        lab.lab_memset(bufs["o"], 0xFF, n * 4)
                        rc = lab.lab_run(v, bufs["w"], bufs["s"], bufs["b"], bufs["xf" if v >= 10 else "x"],
                                         bufs["o"], n, k, grid, 10, 50, ctypes.byref(ms))
                        lab.lab_d2h(out.ctypes.data, bufs["o"], n * 4)
                    if rc != 0:
                        print(f"{label} grid={grid} {VARIANTS[v]} rc={rc}", flush=True)
                        return 3
                    if v != 9:
                        ident.setdefault(v, []).append(bool(np.array_equal(out.view(np.uint32), ref.view(np.uint32))))
                    times[v].append(ms.value)
                post = f"busy={busy()} {cotenants()}"
                print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {label} grid={grid} round={rnd} "
                      f"order={[VARIANTS[v] for v in order]} | pre {pre} | post {post}", flush=True)
            base = statistics.median(times[0]) if 0 in times else None
            for v in variants:
                ms = statistics.median(times[v])
                b = (n * (k // 2)) if v == 9 else nbytes
                row = {"shape": label, "n": n, "k": k, "grid": grid, "variant": VARIANTS[v], "ms": round(ms, 5),
                       "ms_all": [round(t, 5) for t in times[v]], "gb_s": round(b / ms / 1e6, 2),
                       "speedup_vs_prod": round(base / ms, 3) if base else None,
                       "byte_identical": all(ident[v]) if v in ident else None}
                log.write(json.dumps(row) + "\n")
                summary.append(row)
                print(f"  {VARIANTS[v]:15s} {ms:8.4f} ms {row['gb_s']:7.2f} GB/s x{row['speedup_vs_prod']} "
                      f"ident={row['byte_identical']}", flush=True)
        for p in bufs.values():
            lab.lab_free(p)
    bad = [r for r in summary if r["byte_identical"] is False]
    print(f"BYTE-IDENTITY: {'FAIL ' + str([(r['shape'], r['grid'], r['variant']) for r in bad]) if bad else 'all variants identical to prod'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
