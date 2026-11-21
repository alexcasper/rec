"""Which lm_head rows does the K=2048 batched path get wrong, and does it depend on loop iterations?"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, ".")
from mid_mk_driver import f32_to_bf16  # noqa: E402

vp = ctypes.c_void_p


def lib(p):
    lb = ctypes.CDLL(p)
    lb.flint_q4_gemv_host.restype = ctypes.c_int
    lb.flint_q4_gemv_host.argtypes = [vp] * 6 + [ctypes.c_int] * 5 + [vp]
    return lb


ref_l, new_l = lib(sys.argv[1]), lib(sys.argv[2])
rng = np.random.default_rng(42)
n, k = int(sys.argv[3]) if len(sys.argv) > 3 else 150144, 2048
nib = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
s_f = ((rng.random((n, k // 64)) * 0.5 + 0.75) * (0.13 / np.sqrt(k))).astype(np.float32)
b_f = (-7.5 * s_f * (1.0 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32)
x = f32_to_bf16(rng.standard_normal(k).astype(np.float32))
s, b = f32_to_bf16(s_f), f32_to_bf16(b_f)
w = np.ascontiguousarray((nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8))
for grid in (256, 64, (n // 4 + 7) // 8):
    outs = []
    for lb in (ref_l, new_l):
        o = np.zeros(n, np.float32)
        ms = ctypes.c_float()
        assert lb.flint_q4_gemv_host(w.ctypes.data, s.ctypes.data, b.ctypes.data, x.ctypes.data, None,
                                     o.ctypes.data, n, k, grid, 1, 1, ctypes.byref(ms)) == 0
        outs.append(o)
    bad = np.where(outs[0].view(np.uint32) != outs[1].view(np.uint32))[0]
    warps = grid * 8
    it = (bad // 4) // warps  # loop iteration index of the bad rows' quad within its warp
    print(f"grid {grid:5d} ({warps} warps, {-(-n // 4 // warps)} iters/warp): {len(bad)} rows differ; "
          f"rows%4 {np.bincount(bad % 4, minlength=4).tolist()}; iteration hist {np.bincount(it).tolist()[:6]}; "
          f"first {bad[:6].tolist()}")
