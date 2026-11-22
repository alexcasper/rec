"""rec-kcb full-config ppl tier: kernel teacher-forced ppl (ppl__<build>.npz from m3_depth_gate.py --ppl-set) vs the
torch fp32-compute CPU reference at the build's KV tier. Binding at full config: |ppl_rel| < 1e-2 (bf16kv production),
fp32kv <= 1e-4.  PYTHONPATH=src:validation/megakernel_mid python score_ppl_full.py DIR CKPT eval_set_full_v1.json OUT"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk

d, ckpt, evp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ev = json.loads(evp.read_text())
LIM = {"fp32kv": 1e-4, "fp32_bf16kv": 1e-2, "bf16kv": 1e-2}
KV = {"fp32kv": "fp32", "fp32_bf16kv": "bf16", "bf16kv": "bf16"}
ref = mk.Reference(ckpt, "cpu", "fp32")
ref_lp = {}
for kv, dt in (("fp32", None), ("bf16", torch.bfloat16)):
    ref.model.kv_store_dtype = dt
    rows = []
    with torch.inference_mode():
        for it in ev["ppl_items"]:
            t = it["tokens"]
            lg = ref.model.forward(torch.tensor([t]))[0].double()
            rows.append(torch.log_softmax(lg, -1)[:-1].gather(1, torch.tensor(t[1:]).view(-1, 1))[:, 0].numpy())
    ref_lp[kv] = np.asarray(rows)
rep = {"evalset_sha256": ev["sha256"], "builds": {}}
for f in sorted(d.glob("ppl__*.npz")):
    b = f.stem.split("__")[1]
    z = np.load(f)
    assert str(z["evalset_sha256"]) == ev["sha256"], "eval set mismatch"
    klp, rlp = z["logp"], ref_lp[KV[b]]
    nk, nr = float(-klp.mean()), float(-rlp.mean())
    rel = math.exp(nk) / math.exp(nr) - 1.0
    rep["builds"][b] = {"ref_kv": KV[b], "ppl": math.exp(nk), "ref_ppl": math.exp(nr), "ppl_rel_vs_ref": rel,
                        "limit": LIM[b], "pass": abs(rel) < LIM[b], "max_abs_dlogp": float(np.abs(klp - rlp).max()),
                        "per_item_nll": (-klp.mean(1)).tolist(), "ref_per_item_nll": (-rlp.mean(1)).tolist()}
    print(f"ppl {b}: kernel {math.exp(nk):.6g} ref {math.exp(nr):.6g} rel {rel:+.3e} limit {LIM[b]} "
          f"{'PASS' if abs(rel) < LIM[b] else 'FAIL'} max|dlogp| {rep['builds'][b]['max_abs_dlogp']:.3e}")
rep["ref_fp32_vs_bf16_ppl_rel"] = math.exp(float(-ref_lp["bf16"].mean())) / math.exp(float(-ref_lp["fp32"].mean())) - 1
print("reference bf16-KV vs fp32-KV ppl rel", f"{rep['ref_fp32_vs_bf16_ppl_rel']:+.3e}")
out.write_text(json.dumps(rep, indent=1) + "\n")
