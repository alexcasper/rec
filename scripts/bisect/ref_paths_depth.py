"""rec-cnk offline discriminator at depth (no GPU): three views of the same teacher-forced trajectory, all forced to the
kernel's emitted experts (routing removed as a variable):
  R1 = stock reference: one full-sequence forward (prompt + tokens)       [the gate's reference]
  R2 = chunked reference: prompt in CHUNK-sized pieces through KVCache (ring + growing full-attn cache), then each
       decode token one at a time (the kernel's launch structure)
  K  = kernel step_logits / last_logits from the npz
Prints per-step rel(K,R1), rel(K,R2), rel(R1,R2), plus logit scale/margins. Saves rows.
    PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python ref_paths_depth.py NPZ CKPT OUTDIR [CHUNK]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk
from rec_mlx.torch_rocm import KVCache

npz, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
out.mkdir(parents=True, exist_ok=True)
d = np.load(npz)
prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
force = d["step_experts"]
P, N = len(prompt), len(tokens)
K = np.concatenate([d["step_logits"], d["last_logits"][None]], 0).astype(np.float64)   # rows 0..N
t0 = time.time()
ref = mk.Reference(ckpt, "cpu", str(d["kv_dtype"]))
R1, _ = ref.run(prompt, tokens, force=force)
R1 = R1.astype(np.float64)
print(f"R1 done {time.time() - t0:.0f}s", flush=True)
np.save(out / "R1.npy", R1.astype(np.float32))

# R2: chunked through the cache, experts forced with an absolute-position offset
model, cfg = ref.model, ref.model.cfg
real_topk = torch.topk
state = {"off": 0}


def topk(t, k, dim=-1, **kw):
    r = real_topk(t, k, dim=dim, **kw)
    if k != cfg.topk or t.dim() != 3:
        return r
    idx = r.indices.clone()
    o = state["off"]
    idx[0] = torch.as_tensor(force[o:o + idx.shape[1], state["layer"], :], dtype=idx.dtype)
    return torch.return_types.topk((torch.gather(t, dim, idx), idx))


orig_moe = type(model)._moe


def moe(self, x, i):
    state["layer"] = i
    torch.topk = topk
    try:
        return orig_moe(self, x, i)
    finally:
        torch.topk = real_topk


model._moe = moe.__get__(model)
rows = []
seq = prompt + tokens
cache = KVCache(cfg)
with torch.inference_mode():
    i = 0
    while i < P:
        c = seq[i:min(i + CHUNK, P)]
        state["off"] = i
        lg = model.forward(torch.tensor([c]), kv=cache)
        i += len(c)
    rows.append(lg[0, -1].double().numpy())
    for j in range(N):
        state["off"] = P + j
        lg = model.forward(torch.tensor([[seq[P + j]]]), kv=cache)
        rows.append(lg[0, -1].double().numpy())
del model._moe
R2 = np.asarray(rows)
print(f"R2 done {time.time() - t0:.0f}s (chunk {CHUNK})", flush=True)
np.save(out / "R2.npy", R2.astype(np.float32))


def rel(a, b):
    return [float(np.abs(a[i] - b[i]).max() / np.abs(b[i]).max()) for i in range(len(a))]


res = {"npz": str(npz), "P": P, "N": N, "chunk": CHUNK,
       "rel_K_R1": rel(K, R1), "rel_K_R2": rel(K, R2), "rel_R1_R2": rel(R1, R2),
       "absmax_R1": [float(np.abs(r).max()) for r in R1],
       "top2_margin_R1": [float(np.sort(r)[-1] - np.sort(r)[-2]) for r in R1],
       "argmax_K_R1_R2": [[int(K[i].argmax()), int(R1[i].argmax()), int(R2[i].argmax())] for i in range(len(K))]}
for k in ("rel_K_R1", "rel_K_R2", "rel_R1_R2"):
    print(k, "step0 %.3e max %.3e" % (res[k][0], max(res[k])), ["%.1e" % x for x in res[k][:6]])
print("absmax R1 step0..5", ["%.2f" % x for x in res["absmax_R1"][:6]])
(out / "ref_paths.json").write_text(json.dumps(res, indent=1))
