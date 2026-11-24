"""rec-ywy: score kernel ppl emits (fp32kv + int8kv, scripts/m3_depth_gate.py --ppl-set) against the torch fp32-compute reference
with fp32 KV and with the same per-row int8 fake-quant (per-token-head symmetric absmax/127, RNE), teacher-forced.
Implementation check: kernel int8kv vs torch int8 (|dlogp|, top-1 agreement) must look like kernel fp32kv vs torch fp32.
Quality (rec-ywy gate): int8 vs fp32-KV ppl delta and top-1 agreement.
    PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python score_i8.py CKPT EVALSET OUT.json PPL_DIR [PPL_DIR ...]"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk

ckpt, evp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dirs = [Path(d) for d in sys.argv[4:]]
ev = json.loads(evp.read_text())
ref = mk.Reference(ckpt, "cpu", "fp32")
model = ref.model
model.kv_store_dtype = None


def q8_row(t):  # [B, kvh, L, D]: one scale per (token, head) row, matches the kernel's KV-write quantizer
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
    return torch.clamp(torch.round(t / s), -127, 127) * s


def torch_logp(quant):
    model.kv_quant = quant
    lp, t1 = [], []
    with torch.inference_mode():
        for it in ev["ppl_items"]:
            t = it["tokens"]
            lg = model.forward(torch.tensor([t]))[0].double()
            ls = torch.log_softmax(lg, -1)[:-1]
            lp.append(ls.gather(1, torch.tensor(t[1:]).view(-1, 1))[:, 0].numpy())
            t1.append(lg[:-1].argmax(-1).numpy())
    model.kv_quant = None
    return np.asarray(lp), np.asarray(t1)


def ppl(lp):
    return math.exp(float(-lp.mean()))


R = {"fp32": torch_logp(None), "int8": torch_logp(lambda i, k, v: (q8_row(k), q8_row(v)))}
res = {"evalset_sha256": ev["sha256"], "ckpt": str(ckpt),
       "torch": {"int8_vs_fp32": {"ppl_rel": ppl(R["int8"][0]) / ppl(R["fp32"][0]) - 1,
                                  "top1_agree": float((R["int8"][1] == R["fp32"][1]).mean())}}, "kernel": {}}
print("torch int8 vs fp32:", res["torch"]["int8_vs_fp32"], flush=True)
for d in dirs:
    K = {b: np.load(d / f"ppl__{b}.npz") for b in ("fp32kv", "int8kv")}
    row = {}
    for b, rk in (("fp32kv", "fp32"), ("int8kv", "int8")):
        lp, t1 = K[b]["logp"], K[b]["top1"]
        row[f"{b}_vs_torch_{rk}"] = {"max_abs_dlogp": float(np.abs(lp - R[rk][0]).max()),
                                     "ppl_rel": ppl(lp) / ppl(R[rk][0]) - 1, "top1_agree": float((t1 == R[rk][1]).mean())}
    lp8, lp32 = K["int8kv"]["logp"], K["fp32kv"]["logp"]
    row["gate_int8kv_vs_fp32_ref"] = {"ppl_rel": ppl(lp8) / ppl(R["fp32"][0]) - 1,
                                      "top1_agree": float((K["int8kv"]["top1"] == R["fp32"][1]).mean())}
    row["gate_pass"] = abs(row["gate_int8kv_vs_fp32_ref"]["ppl_rel"]) < 0.01 and row["gate_int8kv_vs_fp32_ref"]["top1_agree"] >= 0.97
    res["kernel"][d.name] = row
    print(d.name, json.dumps(row), flush=True)
out.write_text(json.dumps(res, indent=1) + "\n")
