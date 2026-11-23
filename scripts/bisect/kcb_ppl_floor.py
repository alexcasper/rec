"""rec-kcb: ppl noise floor of the torch bf16-KV reference itself on eval_set_full_v1 under fp32-ulp k/v perturbation
(same P(eps) as kcb_sensitivity.py). Compares to the kernel's bf16kv ppl_rel (split +3.38e-3, lop8 +6.92e-3).
    PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python kcb_ppl_floor.py CKPT EVALSET OUT.json [EPS ...]"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk

ckpt, evp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
EPS = [float(e) for e in sys.argv[4:]] or [2.0 ** -23, 2.0 ** -22]
ev = json.loads(evp.read_text())
ref = mk.Reference(ckpt, "cpu", "fp32")
model = ref.model
orig_lin = model._lin


def logp(kv, eps=None, seed=0):
    model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[kv]
    gen = torch.Generator().manual_seed(seed)
    signs = {}

    def lin(name, x):
        y = orig_lin(name, x)
        if eps and name.endswith(("k_proj", "v_proj")):
            key = (name, y.shape[-1])
            if key not in signs:
                signs[key] = torch.randint(0, 2, (y.shape[-2], y.shape[-1]), generator=gen).to(y.dtype) * 2 - 1
            y = y * (1 + eps * signs[key])
        return y

    model._lin = lin
    rows = []
    with torch.inference_mode():
        for it in ev["ppl_items"]:
            t = it["tokens"]
            lg = model.forward(torch.tensor([t]))[0].double()
            rows.append(torch.log_softmax(lg, -1)[:-1].gather(1, torch.tensor(t[1:]).view(-1, 1))[:, 0].numpy())
    model._lin = orig_lin
    return np.asarray(rows)


def ppl(lp):
    return math.exp(float(-lp.mean()))


res = {"evalset_sha256": ev["sha256"], "eps": EPS, "rows": {}}
for kv in ("bf16", "fp32"):
    base = logp(kv)
    res["rows"][f"{kv}_stock"] = {"ppl": ppl(base)}
    for eps in EPS:
        for seed in (1, 2):
            lp = logp(kv, eps, seed)
            r = ppl(lp) / ppl(base) - 1
            res["rows"][f"{kv}_eps{eps:.3g}_s{seed}"] = {"ppl": ppl(lp), "ppl_rel_vs_stock": r,
                                                        "max_abs_dlogp": float(np.abs(lp - base).max())}
            print(f"[{kv} KV] eps {eps:.3g} seed {seed}: ppl rel vs stock {r:+.3e}, max|dlogp| {np.abs(lp - base).max():.3e}",
                  flush=True)
out.write_text(json.dumps(res, indent=1) + "\n")
