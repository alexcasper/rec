"""Numeric validation of the HIP phase-1 megakernel port (rec-tan).

Compiles flint_phases.hip with hipcc (gfx1201), loads it via ctypes, and
checks each translated device function against a torch reference on the
R9700: q4 (INT4-affine matmul), rms, head_norm+RoPE, attention-with-sinks.
"""
import ctypes, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
HIP_SRC = Path(os.environ.get(
    "HIP_SRC", HERE.parent / "native" / "megakernel_hip" / "flint_phases.hip"))
SO = Path(os.environ.get("HIP_SO", HERE / "flint_phases.so"))

GXX = "/nix/store/2ga5nd1m56n5cx2wh8vbf6nrdhqk2f0q-gcc-15.3.0-lib/lib"
DEV = os.environ.get("HIP_TEST_DEVICE", "gfx1201")
TORCH_LIB = sorted(Path(os.environ["REC_TORCH_LIB"]).glob("libamdhip64.so*")) if os.environ.get("REC_TORCH_LIB") else []

if not SO.exists() or SO.stat().st_mtime < HIP_SRC.stat().st_mtime:
    cmd = (
        f"hipcc -std=c++20 --offload-arch={DEV} -O3 -shared -fPIC "
        f"'{HIP_SRC}' -o '{SO}'"
    )
    print("compiling:", cmd, flush=True)
    rc = os.system(cmd)
    assert rc == 0, f"hipcc failed ({rc})"
    print("compiled ->", SO, flush=True)

lib = ctypes.CDLL(str(SO))
for name, restype, argtypes in [
    ("flint_q4", ctypes.c_int, [ctypes.c_void_p]*5 + [ctypes.c_int]*3),
    ("flint_rms", ctypes.c_int, [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_float]),
    ("flint_head_norm", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]),
    ("flint_attn_sinks", ctypes.c_int, [ctypes.c_void_p]*5 + [ctypes.c_int]),
]:
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes

torch.manual_seed(0)
dev = "cuda"
def ptr(t): return ctypes.c_void_p(t.data_ptr())

def ref_quantize(w, group=64, bits=4):
    """MLX affine INT4 packing: nibbles 0..15, dequant = s*nibble + b."""
    shape = w.shape
    wf = w.float().reshape(-1, group)
    lo = wf.min(-1, keepdim=True).values
    hi = wf.max(-1, keepdim=True).values
    s = (hi - lo) / 15.0
    b = lo
    nq = torch.clamp(torch.round((wf - b) / s), 0, 15)
    return nq.reshape(shape), s.reshape(*shape[:-1], shape[-1] // group), b.reshape(*shape[:-1], shape[-1] // group)

# ---- 1. q4: quantized matmul (n=512 rows, K=2048) ----
n, K = 512, 2048
wdq = torch.randn(n, K, device=dev) * 0.02
nq, s, b = ref_quantize(wdq)
# pack nibbles: two 4-bit values per byte, low nibble first
wq_u8 = nq.to(torch.uint8)
packed = (wq_u8[:, 0::2] | (wq_u8[:, 1::2] << 4)).contiguous()
assert packed.shape == (n, K // 2)
# reference uses the DEQUANTIZED quantized weights (the kernel computes with
# s*nibble+b; comparing against original wdq would only measure quant noise)
wdq_ref = (s.float().repeat_interleave(64, dim=1) * nq.float()).reshape(n, K)
x = torch.randn(K, device=dev, dtype=torch.bfloat16)
out = torch.empty(n, device=dev, dtype=torch.bfloat16)
rc = lib.flint_q4(ptr(packed), ptr(s.to(torch.bfloat16).contiguous()),
                 ptr(b.to(torch.bfloat16).contiguous()), ptr(x), ptr(out), n, K, 8)
assert rc == 0, f"flint_q4 rc={rc}"
ref = (wdq_ref @ x.float())
diff = (out.float() - ref).abs().max().item()
rel = diff / ref.abs().max().item()
print(f"q4: maxabs={diff:.4g} rel={rel:.4g} {'PASS' if rel < 2e-3 else 'FAIL'}")

# ---- 2. rms: hidden 2048 ----
hidden, eps = 2048, 1e-6
xh = torch.randn(hidden, device=dev, dtype=torch.bfloat16)
wh = torch.rand(hidden, device=dev, dtype=torch.bfloat16) + 0.5
outr = torch.empty_like(xh)
rc = lib.flint_rms(ptr(xh), ptr(wh), ptr(outr), hidden, eps)
assert rc == 0, f"flint_rms rc={rc}"
refr = (xh.float() * torch.rsqrt(xh.float().pow(2).mean() + eps) * wh.float())
diffr = (outr.float() - refr).abs().max().item()
print(f"rms: maxabs={diffr:.4g} {'PASS' if diffr < 1e-1 else 'FAIL'}")

# ---- 3. head_norm + RoPE (32 heads, head_dim 128, position 17) ----
heads = 32
xn = torch.randn(heads * 128, device=dev, dtype=torch.bfloat16)
wn = torch.rand(128, device=dev, dtype=torch.bfloat16) + 0.5
logbase = torch.log2(torch.tensor(10000.0)).item()
pos = 17
ref_head = xn.clone().reshape(heads, 128).float()
inv = torch.rsqrt(ref_head.pow(2).mean(-1, keepdim=True) + eps) * wn.float()
normed = ref_head * inv
pairs = torch.arange(64, device=dev)
theta = pos * torch.exp2(-pairs.float() / 64.0 * logbase)
c, si = torch.cos(theta), torch.sin(theta)
a, bb = normed[:, :64].clone(), normed[:, 64:].clone()
normed = torch.cat([a * c - bb * si, a * si + bb * c], dim=-1)
rc = lib.flint_head_norm(ptr(xn), ptr(wn), heads, 1, pos, logbase, eps)
assert rc == 0, f"flint_head_norm rc={rc}"
diffn = (xn.reshape(heads, 128).float() - normed).abs().max().item()
print(f"head_norm+rope: maxabs={diffn:.4g} {'PASS' if diffn < 5e-1 else 'FAIL'}")

# ---- 4. attention with sinks (4 kv groups x 8 q-heads, head_dim 128) ----
count = 129
nheads, nkv = 32, 4
q = torch.randn(nheads * 128, device=dev, dtype=torch.bfloat16)
sinks = torch.randn(nheads, device=dev, dtype=torch.bfloat16) * 0.5
k = torch.randn(nkv * count * 128, device=dev, dtype=torch.bfloat16)
v = torch.randn(nkv * count * 128, device=dev, dtype=torch.bfloat16)
outa = torch.empty(nheads * 128, device=dev, dtype=torch.bfloat16)
rc = lib.flint_attn_sinks(ptr(q), ptr(k), ptr(v), ptr(sinks), ptr(outa), count)
assert rc == 0, f"flint_attn_sinks rc={rc}"
# reference: per kv group, per q head: scores incl sink, softmax, weighted v
qr = q.reshape(nheads, 128).float() * 0.08838834764831845
refa = torch.empty(nheads, 128, device=dev)
for kv in range(nkv):
    kr = k.reshape(nkv, count, 128)[kv].float()
    vr = v.reshape(nkv, count, 128)[kv].float()
    for g in range(8):
        head = kv * 8 + g
        scores = qr[head] @ kr.T  # [count]
        scores = torch.cat([scores, sinks.float()[head].unsqueeze(0)])
        probs = torch.softmax(scores, dim=0)[:count]
        refa[head] = probs @ vr
diffa = (outa.reshape(nheads, 128).float() - refa).abs().max().item()
print(f"attn_sinks: maxabs={diffa:.4g} {'PASS' if diffa < 5e-2 else 'FAIL'}")

print("HIP phase-1 validation complete.")
