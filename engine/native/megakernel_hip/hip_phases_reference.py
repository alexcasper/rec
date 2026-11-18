"""Torch-side references for the HIP phase-1 outputs (separate process, so
torch's ROCm runtime never mixes with the kernel process's runtime)."""
import sys
import numpy as np
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "hip_phases_io.npz")

def bits_to_float(u16):
    # bf16 bits -> float32: place in the high 16 bits of a uint32, then view
    return torch.from_numpy((u16.copy().astype(np.uint32) << 16)).view(torch.float32)

def tobf(x):
    return x.to(torch.bfloat16).float()

fails = []

# ---- q4 ----
s = bits_to_float(d["q4_s_bits"]); b = bits_to_float(d["q4_b_bits"])
xb = bits_to_float(d["q4_x_bits"])
packed = torch.from_numpy(d["q4_w_packed"].copy()).reshape(512, 1024)
# decode nibbles: byte i -> low = element 2i, high = element 2i+1
lo = (packed & 0xF).to(torch.float32); hi = ((packed >> 4) & 0xF).to(torch.float32)
nq = torch.stack([lo, hi], dim=-1).reshape(512, 2048)
sfull = s.repeat_interleave(64, dim=1); bfull = b.repeat_interleave(64, dim=1)
deq = sfull * nq + bfull
xref = xb.clone()  # kernel consumed bf16-rounded x
ref = deq @ xref.clone().float()
out = bits_to_float(d["q4_out_bits"])
diff = (out - ref).abs().max().item()
print(f"q4: maxabs={diff:.4g} rel={diff/ref.abs().max().item():.4g}")
if diff / ref.abs().max().item() > 2e-3: fails.append("q4")

# ---- rms ----
x = torch.from_numpy(d["rms_x"]); w = torch.from_numpy(d["rms_w"])
xk = tobf(x)  # kernel saw bf16-rounded inputs
ref = xk.float() * torch.rsqrt(xk.float().pow(2).mean() + 1e-6) * tobf(w).float()
out = bits_to_float(d["rms_out_bits"])
diff = (out - ref).abs().max().item()
print(f"rms: maxabs={diff:.4g}")
if diff > 0.05: fails.append("rms")

# ---- head_norm ----
heads = 32; eps = 1e-6
logbase = float(d["hn_logbase"]); pos = int(d["hn_pos"])
xk = bits_to_float(d["hn_x_bits"]).reshape(heads, 128)
w = bits_to_float(d["hn_w_bits"])
inv = torch.rsqrt(xk.pow(2).mean(-1, keepdim=True) + eps) * w
normed = xk * inv
pairs = torch.arange(64, dtype=torch.float32)
theta = pos * torch.exp2(-pairs / 64.0 * logbase)
c, si = torch.cos(theta), torch.sin(theta)
a, bb = normed[:, :64], normed[:, 64:]
ref = torch.cat([a * c - bb * si, a * si + bb * c], dim=-1)
out = bits_to_float(d["hn_x_bits"]).reshape(heads, 128)
diff = (out - ref).abs().max().item()
print(f"head_norm: maxabs={diff:.4g}")
if diff > 1.0: fails.append("head_norm")

# ---- attention sinks ----
nheads, nkv, count = 32, 4, 129
q = bits_to_float(d["attn_q"]).reshape(nheads, 128) * 0.08838834764831845
sinks = bits_to_float(d["attn_sinks"])
k = bits_to_float(d["attn_k"]).reshape(nkv, count, 128)
v = bits_to_float(d["attn_v"]).reshape(nkv, count, 128)
out = bits_to_float(d["attn_out_bits"]).reshape(nheads, 128)
refa = torch.empty(nheads, 128)
for kv in range(nkv):
    for g in range(8):
        head = kv * 8 + g
        sc = torch.cat([q[head] @ k[kv].T, sinks[head].unsqueeze(0)])
        refa[head] = torch.softmax(sc, 0)[:count] @ v[kv]
diff = (out - refa).abs().max().item()
print(f"attn_sinks: maxabs={diff:.4g}")
if diff > 0.2: fails.append("attn_sinks")

print("FAILURES:", fails if fails else "none — ALL PHASES PASS")
sys.exit(1 if fails else 0)
