"""rec-3cr: compare full-config wrap512 emits across KV modes (no reference): where does bf16-KV diverge from fp32-KV?"""
import sys

import numpy as np

g = sys.argv[1]
a = np.load(f"{g}/wrap512_fp32kv.npz")
b = np.load(f"{g}/wrap512_fp32_bf16kv.npz")
c = np.load(f"{g}/wrap512_bf16kv.npz")
for n, z in (("fp32kv", a), ("fp32_bf16kv", b), ("bf16kv", c)):
    print(f"tokens {n:12s}", z["tokens"].tolist()[:12])
sa, sb = a["step_logits"], b["step_logits"]
for i in range(6):
    r = np.abs(sa[i] - sb[i]).max() / np.abs(sa[i]).max()
    srt = np.sort(sa[i])
    print(f"step {i}: rel(fp32kv vs fp32_bf16kv) {r:.3e} same-prefix {a['tokens'][:i].tolist() == b['tokens'][:i].tolist()} "
          f"top2 gap fp32kv {srt[-1] - srt[-2]:.3f}")
ea, eb = a["step_experts"], b["step_experts"]
P = len(a["prompt"])
d = (np.sort(ea[:P], -1) != np.sort(eb[:P], -1)).any(-1)
pos = np.nonzero(d.any(1))[0]
print("prompt positions with any routing difference fp32kv vs fp32_bf16kv:", len(pos), "first:", pos[:12].tolist(), "of", P)
print("per-64-position bucket counts:", [int(d[i:i + 64].any(1).sum()) for i in range(0, P, 64)])
lay = np.nonzero(d.any(0))[0]
print("layers with any difference:", lay.tolist()[:20], "first diff layer at first pos:",
      np.nonzero(d[pos[0]])[0].tolist() if len(pos) else None)
