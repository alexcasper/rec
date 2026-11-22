"""rec-kcb depth question: is the fp32-KV 1e-4 tier at depth 8192 sensitive to the RoPE inv_freq convention?
Re-score a kernel npz with the reference's rope_neox patched to the KERNEL's formula
(inv = exp2f(-(pair/64) * log2(10000)) in fp32, angle = (float)pos * inv), everything else unchanged.
    PYTHONPATH=src:validation/megakernel_mid python rope_mimic_score.py NPZ CKPT OUT.json"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import rec_mlx.torch_rocm as tr
import mid_mk_check as mk

LOGBASE = torch.tensor(13.28771237954945, dtype=torch.float32)


def rope_kernel_style(x, offset, theta):
    _, _, L, D = x.shape
    half = D // 2
    pos = torch.arange(L, device=x.device, dtype=torch.float32) + offset
    inv = torch.exp2((-(torch.arange(half, dtype=torch.float32)) / 64.0) * LOGBASE)
    freqs = torch.outer(pos, inv)
    c = freqs.cos()[None, None].to(x.dtype)
    s = freqs.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


npz, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
inv_stock = torch.pow(10000.0, -torch.arange(64, dtype=torch.float32) / 64)
inv_kern = torch.exp2((-(torch.arange(64, dtype=torch.float32)) / 64.0) * LOGBASE)
print("inv_freq differing pairs:", int((inv_stock != inv_kern).sum()), "max rel", float(((inv_stock - inv_kern) / inv_stock).abs().max()))
tr.rope_neox = rope_kernel_style
d = np.load(npz)
kv = str(d["kv_dtype"])
ref = mk.Reference(ckpt, "cpu", kv)
r = mk.check_arrays(ref, d, d["prompt"].tolist(), d["tokens"].tolist(), kv, mk.vocab_of(ckpt), str(d["case"]))
r.pop("_ref", None)
row = {k: r.get(k) for k in ("pass", "problems", "last_logits_rel", "step_logits_max_rel", "n_routing_deviations", "tokens_exact")}
row["rope"] = "kernel-style exp2f inv_freq"
print(json.dumps(row))
out.write_text(json.dumps(row, indent=1))
