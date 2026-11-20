"""rec-6ob step 4 gate + per-op timing: variant .so vs base (2fd83cb) .so via flint_q4_gemv_host.
Gate: 7 shapes (every K path); base output once per shape; variant 20 runs at grid 0 + 20 at grid 16,
every run bitwise-equal to base. On divergence, report the first differing rows and rel/ulp size.
Timing (only if --time and the gate passed): o_proj, expert_down, lm_head, A9 method (warmup 10,
50 launches), ABBA base/variant x ROUNDS, busy-gated, medians.
    python gate7.py base.so var.so [--time] [--rounds 6]"""
import argparse
import ctypes
import statistics
import sys

import numpy as np

sys.path.insert(0, ".")
from mid_mk_driver import f32_to_bf16  # noqa: E402
from q4_lab import busy, cotenants, gate  # noqa: E402

vp = ctypes.c_void_p
SHAPES = {"expert_gate_512x2048": (512, 2048), "expert_down_2048x512": (2048, 512),
          "shared_down_2048x1024": (2048, 1024), "shared_gate_1024x2048": (1024, 2048),
          "q_proj_4096x2048": (4096, 2048), "o_proj_2048x4096": (2048, 4096), "lm_head_150144x2048": (150144, 2048)}
TIMED = ("o_proj_2048x4096", "expert_down_2048x512", "lm_head_150144x2048")


def load(p):
    lb = ctypes.CDLL(p)
    lb.flint_q4_gemv_host.restype = ctypes.c_int
    lb.flint_q4_gemv_host.argtypes = [vp] * 6 + [ctypes.c_int] * 5 + [vp]
    return lb


def call(lb, d, n, k, grid, warm, iters):
    o = np.zeros(n, np.float32)
    ms = ctypes.c_float()
    rc = lb.flint_q4_gemv_host(d["w"].ctypes.data, d["s"].ctypes.data, d["b"].ctypes.data, d["x"].ctypes.data, None,
                               o.ctypes.data, n, k, grid, warm, iters, ctypes.byref(ms))
    assert rc == 0, rc
    return o, ms.value


def data(n, k):
    rng = np.random.default_rng(42)
    s_f = ((rng.random((n, k // 64)) * 0.5 + 0.75) * (0.13 / np.sqrt(k))).astype(np.float32)
    return {"w": rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8), "s": f32_to_bf16(s_f),
            "b": f32_to_bf16((-7.5 * s_f * (1.0 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32)),
            "x": f32_to_bf16(rng.standard_normal(k).astype(np.float32))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("var")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    lb, lv = load(a.base), load(a.var)
    ok_all = True
    datas = {}
    for name, (n, k) in SHAPES.items():
        d = datas[name] = data(n, k)
        ref, _ = call(lb, d, n, k, 0, 0, 1)
        r32 = ref.view(np.uint32)
        bad_runs, first_bad = 0, None
        for grid in (0, 16):
            for _ in range(20):
                o, _ = call(lv, d, n, k, grid, 0, 1)
                if not np.array_equal(o.view(np.uint32), r32):
                    bad_runs += 1
                    if first_bad is None:
                        idx = np.where(o.view(np.uint32) != r32)[0]
                        ulp = np.abs(o.view(np.int32)[idx].astype(np.int64) - ref.view(np.int32)[idx])
                        first_bad = (len(idx), idx[:4].tolist(), float(np.abs(o - ref).max() / np.abs(ref).max()),
                                     int(ulp.max()))
        ok = bad_runs == 0
        ok_all &= ok
        msg = "IDENTICAL x40" if ok else (f"DIFF in {bad_runs}/40 runs: {first_bad[0]} rows e.g. {first_bad[1]}, "
                                          f"max rel {first_bad[2]:.2e}, max {first_bad[3]} ulp")
        print(f"  {name:24s} {msg}", flush=True)
    print("GATE byte-identity vs base:", "PASS" if ok_all else "FAIL", flush=True)
    if not (a.time and ok_all):
        return 0 if ok_all else 1
    times = {(nm, t): [] for nm in TIMED for t in "BV"}
    for rnd in range(a.rounds):
        if not gate():
            print("GATE TIMEOUT")
            return 2
        pre = f"busy={busy()} {cotenants()}"
        for t in ("BV" if rnd % 2 == 0 else "VB"):
            for nm in TIMED:
                n, k = SHAPES[nm]
                times[(nm, t)].append(call(lb if t == "B" else lv, datas[nm], n, k, 16, 10, 50)[1])
        print(f"  round {rnd} {'BV' if rnd % 2 == 0 else 'VB'} | pre {pre} | post busy={busy()} {cotenants()}", flush=True)
    for nm in TIMED:
        b, v = statistics.median(times[(nm, "B")]), statistics.median(times[(nm, "V")])
        print(f"  TIME {nm:24s} grid16 base {b:.4f} ms  var {v:.4f} ms  speedup {b / v:.3f}  "
              f"(var range {min(times[(nm, 'V')]):.4f}-{max(times[(nm, 'V')]):.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
