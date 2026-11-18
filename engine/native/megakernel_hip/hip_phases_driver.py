"""Generate tiny-Flint phase inputs with numpy RNG (seed 0) and run the HIP
host wrappers in a torch-free process (single ROCm runtime). Saves an npz with
inputs + kernel outputs for the torch reference script."""
import ctypes, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "hip_phases_io.npz"
rng = np.random.default_rng(0)
bf16 = np.uint16  # bf16 transported as raw uint16 bits

results = {}

# ---- inputs ----
wdq = (rng.standard_normal((512, 2048)) * 0.02).astype(np.float32)
wf = wdq.reshape(-1, 64)
lo = wf.min(-1); hi = wf.max(-1)
s32 = ((hi - lo) / 15.0).reshape(512, 32)
b32 = lo.reshape(512, 32)
# float32 -> bf16 bits: reuse torch-free rounding via numpy float32 truncate
def f32_to_bf16(v):
    u = v.astype(np.float32).view(np.uint32)
    return ((u + 0x8000) >> 16).astype(np.uint16)  # round-to-nearest-even-ish
s16 = f32_to_bf16(s32); b16 = f32_to_bf16(b32)
nq = np.clip(np.round((wf - b32.reshape(-1, 1)) / s32.reshape(-1, 1)), 0, 15).reshape(512, 2048).astype(np.uint8)
packed8 = (nq[:, 0::2] | (nq[:, 1::2] << 4)).astype(np.uint8)  # [512, K/2] bytes
packed_bytes = packed8.tobytes()
packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
x = (rng.standard_normal(2048) * 1.0).astype(np.float32)

lib = ctypes.CDLL("/home/albion/rec-ve/flint_phases.so")
def bind(name, n_ptr, ints=(), floats=()):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * n_ptr + [ctypes.c_int] * len(ints) + [ctypes.c_float] * len(floats)
    return fn
lib.flint_q4_host = bind("flint_q4_host", 5, ints=(0, 0, 0))
lib.flint_rms_host = bind("flint_rms_host", 3, ints=(0,), floats=(0,))
lib.flint_head_norm_host = bind("flint_head_norm_host", 2, ints=(0, 0, 0), floats=(0, 0))
lib.flint_attn_sinks_host = bind("flint_attn_sinks_host", 5, ints=(0,))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"  # q4 | rms | norm | attn | all
# ---- q4 ---- (hold bf16 arrays alive: .ctypes.data of a temporary dangles)
xb16 = f32_to_bf16(x)
out_q4 = np.zeros(512, dtype=np.uint16)
rc = lib.flint_q4_host(packed_arr.ctypes.data, s16.ctypes.data, b16.ctypes.data,
                       xb16.ctypes.data, out_q4.ctypes.data, 512, 2048, 8)
print("q4 rc", rc)
results["q4_w_packed"] = packed_arr
results["q4_s_bits"] = s16; results["q4_b_bits"] = b16
results["q4_x_bits"] = xb16
results["q4_out_bits"] = out_q4

# ---- rms ----
hidden = 2048; eps = 1e-6
xr = rng.standard_normal(hidden).astype(np.float32)
wr = (rng.random(hidden) + 0.5).astype(np.float32)
outr = np.zeros(hidden, dtype=np.uint16)
xr16 = f32_to_bf16(xr); wr16 = f32_to_bf16(wr)
rc = lib.flint_rms_host(xr16.ctypes.data, wr16.ctypes.data,
                        outr.ctypes.data, hidden, eps)
print("rms rc", rc)
results["rms_x"] = xr; results["rms_w"] = wr
results["rms_out_bits"] = outr

# ---- head_norm + rope ----
if STAGE in ("norm", "attn", "all"):
    heads = 32
    xn = rng.standard_normal(heads * 128).astype(np.float32)
    wn = (rng.random(128) + 0.5).astype(np.float32)
    logbase = float(np.log2(10000.0)); pos = 17
    xh = f32_to_bf16(xn); wh16 = f32_to_bf16(wn)
    xh_in = xh.copy()
    rc = lib.flint_head_norm_host(xh.ctypes.data, wh16.ctypes.data,
                                  heads, 1, pos, logbase, eps)
    print("head_norm rc", rc)
    results["hn_x_in_bits"] = xh_in; results["hn_x_out_bits"] = xh
    results["hn_w_bits"] = wh16
    results["hn_logbase"] = np.float32(logbase); results["hn_pos"] = np.int32(pos)

# ---- attention with sinks ----
if STAGE in ("attn", "all"):
    nheads, nkv, count = 32, 4, 129
    q = rng.standard_normal(nheads * 128).astype(np.float32)
    sinks = (rng.standard_normal(nheads) * 0.5).astype(np.float32)
    kk = rng.standard_normal(nkv * count * 128).astype(np.float32)
    vv = rng.standard_normal(nkv * count * 128).astype(np.float32)
    outa = np.zeros(nheads * 128, dtype=np.uint16)
    q16 = f32_to_bf16(q); k16 = f32_to_bf16(kk); v16 = f32_to_bf16(vv); sn16 = f32_to_bf16(sinks)
    rc = lib.flint_attn_sinks_host(q16.ctypes.data, k16.ctypes.data, v16.ctypes.data,
                                   sn16.ctypes.data, outa.ctypes.data, count)
    print("attn rc", rc)
    results["attn_q"] = q; results["attn_k"] = kk; results["attn_v"] = vv; results["attn_sinks"] = sinks
    results["attn_out_bits"] = outa

np.savez(OUT, **results)
print("saved", OUT, len(results), "arrays")
