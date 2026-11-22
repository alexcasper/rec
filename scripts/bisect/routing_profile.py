"""rec-cnk: per-position routing profile of a depth emit vs the torch fp32 reference forced to the kernel's routing.
Lists every (layer, pos) where the kernel's expert set differs from the reference's own top-k there, with the
reference selection-score gap (k-th vs k+1-th) and whether it is a k/(k+1) exchange; plus the gap distribution
(how many near-ties exist per depth bucket). Forced-routing rows are compared to the kernel logits.
    PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python routing_profile.py NPZ CKPT OUT.json"""
import json
import sys
from pathlib import Path

import numpy as np

import mid_mk_check as mk

npz, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
d = np.load(npz)
prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
ref = mk.Reference(ckpt, "cpu", str(d["kv_dtype"]))
dev, ref_f = mk.routing_deviations(ref, prompt, tokens, d["step_experts"])
K = np.concatenate([d["step_logits"], d["last_logits"][None]], 0).astype(np.float64)
rows = [float(np.abs(K[i] - ref_f[i]).max() / np.abs(ref_f[i]).max()) for i in range(len(K))]
res = {"npz": str(npz), "positions": len(prompt) + len(tokens), "n_deviations": len(dev), "deviations": dev,
       "forced_rel_per_row": rows, "forced_rel_max": max(rows),
       "forced_rows_over_1e-4": [i for i, r in enumerate(rows) if r > 1e-4]}
print(json.dumps({k: v for k, v in res.items() if k != "forced_rel_per_row"}, indent=1))
out.write_text(json.dumps(res, indent=1))
