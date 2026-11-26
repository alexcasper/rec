"""rec-21a step-2 timing probe runner (dusk, gfx1150): P0 dequant-FMA baseline vs P1 dot4+unpack vs P3 quantize pass.
Compiles w4a8_probe.hip (hipcc, hip_phases_test convention), checks correctness vs host references,
times with hipEvents (min of 3 runs of REPS launches). Prints a JSON verdict line.
    python w4a8_probe_run.py [--rows 4096] [--reps 50]
"""
import ctypes, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC, SO = HERE / "w4a8_probe.hip", HERE / "w4a8_probe.so"
rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 4096
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 50
K, ROWB, NG64, NG16, NA4 = 2048, 1024, 32, 128, 512

if not SO.exists() or SO.stat().st_mtime < SRC.stat().st_mtime:
    cmd = f"hipcc -std=c++20 --offload-arch=gfx1150 -O3 -shared -fPIC '{SRC}' -o '{SO}'"
    print("compiling:", cmd, flush=True)
    assert os.system(cmd) == 0, "hipcc failed"
    print("compiled ->", SO, flush=True)

hip = ctypes.CDLL("libamdhip64.so")
hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
so = ctypes.CDLL(str(SO))
for name, nargs in (("p0_dequant_fma", 6), ("p1_dot4_unpack", 6), ("p3_quantize", 4)):
    fn = getattr(so, name); fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * (nargs - 1) + [ctypes.c_int]
V = ctypes.c_void_p

def dmem(a):
    p = V()
    assert hip.hipMalloc(ctypes.byref(p), a.nbytes) == 0
    if a.dtype.hasobject is False:
        assert hip.hipMemcpy(p, a.ctypes.data, a.nbytes, 0) == 0  # H2D
    return p

def dget(p, like):
    out = np.empty_like(like)
    assert hip.hipMemcpy(out.ctypes.data, p, out.nbytes, 2) == 0  # D2H
    return out

ev0, ev1 = V(), V()
assert hip.hipEventCreate(ctypes.byref(ev0)) == 0 and hip.hipEventCreate(ctypes.byref(ev1)) == 0

rng = np.random.default_rng(2049)
w = rng.integers(0, 16, size=(rows, ROWB), dtype=np.uint8)
def to_bf16_bits(v):  # truncate fp32 -> bf16 bits (device decodes exactly these)
    return (v.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
s_bits = to_bf16_bits(rng.random((rows, NG64)).astype(np.float32) * 1.0 + 0.5)
b_bits = to_bf16_bits((rng.random((rows, NG64)).astype(np.float32) - 0.5) * 0.01)
def from_bf16_bits(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)
s_h, b_h = from_bf16_bits(s_bits), from_bf16_bits(b_bits)
x = rng.standard_normal(K).astype(np.float32)

# host references (float64) — g64-affine: w_deq[k] = sv[k//64]*nib[k] + bv[k//64]
wq = np.stack([w & 0xF, w >> 4], axis=-1).reshape(rows, K).astype(np.float64)  # deinterleave nibbles
xd = x.astype(np.float64)
sve = np.repeat(s_h.astype(np.float64), 64, axis=1)
bve = np.repeat(b_h.astype(np.float64), 64, axis=1)
y_ref = ((sve * wq + bve) @ xd)

# host quantizer (matches P3): per-g16 absmax/RNE -> s8 + scales, packed A-dwords
qa = np.zeros(K, dtype=np.int8); sa = np.zeros(NG16, dtype=np.float32)
for g16 in range(NG16):
    seg = x[g16 * 16:(g16 + 1) * 16]
    sc = max(float(np.abs(seg).max()), 1e-30) / 127.0
    qa[g16 * 16:(g16 + 1) * 16] = np.rint(seg / sc).clip(-127, 127).astype(np.int8)
    sa[g16] = sc
qa4 = np.ascontiguousarray(qa.view(np.int32))
xq = (sa.repeat(16).astype(np.float64) * qa.astype(np.float64))
y_ref_q = ((sve * wq + bve) @ xq)

dw, ds, db, dx = dmem(w), dmem(s_bits), dmem(b_bits), dmem(x)
dy = dmem(np.zeros(rows, np.float32))
dqa4, dsa = dmem(qa4), dmem(sa)

def timed(fn, nbuf, like):
    times = []
    for _ in range(3):
        assert hip.hipEventRecord(ev0, None) == 0
        for _ in range(reps):
            fn()
        assert hip.hipEventRecord(ev1, None) == 0 and hip.hipDeviceSynchronize() == 0
        ms = ctypes.c_float()
        assert hip.hipEventElapsedTime(ctypes.byref(ms), ev0, ev1) == 0
        times.append(ms.value)
    return min(times), dget(nbuf, like)

assert so.p0_dequant_fma(dw, ds, db, dx, dy, rows) == 0
t0, y0 = timed(lambda: so.p0_dequant_fma(dw, ds, db, dx, dy, rows), dy, np.zeros(rows, np.float32))
r0 = float(np.abs(y0.astype(np.float64) - y_ref).max() / np.abs(y_ref).max())

assert so.p1_dot4_unpack(dw, ds, db, dqa4, dsa, dy, rows) == 0
t1, y1 = timed(lambda: so.p1_dot4_unpack(dw, ds, db, dqa4, dsa, dy, rows), dy, np.zeros(rows, np.float32))
r1 = float(np.abs(y1.astype(np.float64) - y_ref_q).max() / np.abs(y_ref_q).max())

nvec = 512
xv = rng.standard_normal((nvec, K)).astype(np.float32)
dxv = dmem(xv)
dqa4v = dmem(np.zeros(nvec * NA4, np.uint32))
dsav = dmem(np.zeros(nvec * NG16, np.float32))
assert so.p3_quantize(dxv, dqa4v, dsav, nvec) == 0
t3, _ = timed(lambda: so.p3_quantize(dxv, dqa4v, dsav, nvec), dqa4v, np.zeros(nvec * NA4, np.uint32))
qg = dget(dqa4v, np.zeros(nvec * NA4, np.uint32)).view(np.int8).reshape(nvec, K).astype(np.float64)
sg = dget(dsav, np.zeros(nvec * NG16, np.float32)).reshape(nvec, NG16).repeat(16, 1).astype(np.float64)
r3 = float(np.abs(sg * qg - xv.astype(np.float64)).max())

print(json.dumps({
    "rows": rows, "reps": reps, "K": K,
    "p0_dequant_fma": {"ms_per_launch": round(t0 / reps, 4), "rel_err_max": r0, "us_per_gemv": round(t0 / reps * 1e3, 2)},
    "p1_dot4_unpack": {"ms_per_launch": round(t1 / reps, 4), "rel_err_max": r1,
                       "us_per_gemv": round(t1 / reps * 1e3, 2), "speedup_vs_p0": round(t0 / t1, 3)},
    "p3_quantize": {"ms_per_512_vecs": round(t3 / reps, 4), "us_per_vec": round(t3 / reps / nvec * 1e3, 2),
                    "max_dequant_abs_err": r3},
    "note": "P1 uses conservative scalar nibble unpack; vectorized perm tricks only improve it. "
            "P2 (pre-unpacked bank) infeasible: 2x 21GiB bank > VRAM. dot8_i32_iu4 likely needs i4 "
            "activations (fails quality per rec-hyf) - open question for sunny."
}, indent=1))
