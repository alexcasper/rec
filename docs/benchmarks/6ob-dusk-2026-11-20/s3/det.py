"""rec-6ob determinism + accuracy check of flint_q4_gemv_host across all gemv K paths.
Each shape: one input set (seed 42), REPS launches at two grids; all outputs must be bitwise equal to
each other (nondeterminism hunt: the 2-rows-in-flight K=2048 variant failed lm_head 3/6 runs) and
within rel 1e-6 of an fp64 reference.   python det.py <lib.so> [reps]"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, ".")
from mid_mk_driver import bf16_to_f32, f32_to_bf16

vp = ctypes.c_void_p
lib = ctypes.CDLL(sys.argv[1])
lib.flint_q4_gemv_host.restype = ctypes.c_int
lib.flint_q4_gemv_host.argtypes = [vp] * 6 + [ctypes.c_int] * 5 + [vp]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SHAPES = [(512, 2048), (2048, 512), (2048, 1024), (1024, 2048), (4096, 2048), (2048, 4096), (150144, 2048)]
bad_total = 0
for n, k in SHAPES:
    rng = np.random.default_rng(42)
    nib = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    s_f = ((rng.random((n, k // 64)) * 0.5 + 0.75) * (0.13 / np.sqrt(k))).astype(np.float32)
    b_f = (-7.5 * s_f * (1.0 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32)
    x = f32_to_bf16(rng.standard_normal(k).astype(np.float32))
    s, b = f32_to_bf16(s_f), f32_to_bf16(b_f)
    w = np.ascontiguousarray((nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8))
    ref = ((np.repeat(bf16_to_f32(s).astype(np.float64), 64, 1) * nib + np.repeat(bf16_to_f32(b).astype(np.float64), 64, 1))
           @ bf16_to_f32(x).astype(np.float64))
    del nib
    outs = []
    for grid in (0, 16):
        for _ in range(REPS):
            o = np.zeros(n, np.float32)
            ms = ctypes.c_float()
            assert lib.flint_q4_gemv_host(w.ctypes.data, s.ctypes.data, b.ctypes.data, x.ctypes.data, None,
                                          o.ctypes.data, n, k, grid, 0, 1, ctypes.byref(ms)) == 0
            outs.append(o)
    first = outs[0].view(np.uint32)
    ndiff = sum(not np.array_equal(first, o.view(np.uint32)) for o in outs[1:])
    rel = max(float(np.abs(o - ref).max() / np.abs(ref).max()) for o in outs)
    ok = ndiff == 0 and rel < 1e-6
    bad_total += not ok
    print(f"{n:6d}x{k:<5d} K-path {'2048' if k == 2048 else 'generic'}: {len(outs)} runs, {ndiff} differ from run 0, "
          f"max rel vs fp64 {rel:.2e} -> {'OK' if ok else 'FAIL'}", flush=True)
print("DETERMINISM+ACCURACY:", "PASS" if not bad_total else f"FAIL ({bad_total} shapes)")
