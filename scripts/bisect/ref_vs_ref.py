"""rec-kcb: reference-vs-reference at full config: torch fp32-KV vs torch bf16-KV (both fp32 compute), teacher-forced on
the same wrap512 trajectory. If this alone exceeds the 1e-3 tier, the bf16-KV tier is unattainable at full Flint by any kernel."""
import json
import sys
from pathlib import Path

import numpy as np

import mid_mk_check as mk

npz, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(npz)
prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
r32, _ = mk.Reference(ckpt, "cpu", "fp32").run(prompt, tokens)
r16, _ = mk.Reference(ckpt, "cpu", "bf16").run(prompt, tokens)
rel = [float(np.abs(r32[i] - r16[i]).max() / np.abs(r32[i]).max()) for i in range(len(r32))]
argmax_diff = [i for i in range(len(r32)) if int(r32[i].argmax()) != int(r16[i].argmax())]
out = {"trajectory": str(npz.name), "rows": len(rel), "rel_max": max(rel), "rel_first": rel[0], "rel_last": rel[-1],
       "rel_per_step": [round(x, 6) for x in rel], "argmax_differs_at": argmax_diff}
print(json.dumps(out))
