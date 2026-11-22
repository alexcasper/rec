"""rec-e5d gate scoring (dawn) for one mid candidate's gate dir: wrap512 x 3 KV builds (strict spec (a) + forced routing),
depth8192 fp32kv (strict + forced, with routing deviations), ppl eval_set_v1 + eval_set_full_v1 (vs the torch reference).
    PYTHONPATH=src:validation/megakernel_mid python score_e5d.py GATEDIR CKPT OUT.json"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk

g, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
vocab = mk.vocab_of(ckpt)
refs = {}


def R(kv):
    if kv not in refs:
        refs[kv] = mk.Reference(ckpt, "cpu", kv)
    return refs[kv]


def score(npz):
    d = np.load(npz)
    kv = str(d["kv_dtype"])
    ref = R(kv)
    r = mk.check_arrays(ref, d, d["prompt"].tolist(), d["tokens"].tolist(), kv, vocab, str(d["case"]))
    r.pop("_ref", None)
    dev, ref_f = mk.routing_deviations(ref, d["prompt"].tolist(), d["tokens"].tolist(), d["step_experts"])
    K = np.concatenate([d["step_logits"], d["last_logits"][None]], 0).astype(np.float64)
    forced = max(float(np.abs(K[i] - ref_f[i]).max() / np.abs(ref_f[i]).max()) for i in range(len(K)))
    return {"kv": kv, "tier": mk.LOGIT_RTOL[kv], "strict_pass": r["pass"], "strict_last": r["last_logits_rel"],
            "strict_step": r["step_logits_max_rel"], "problems": r["problems"], "forced_max": forced,
            "forced_pass": forced <= mk.LOGIT_RTOL[kv], "n_dev": len(dev),
            "devs": [{k: x[k] for k in ("layer", "pos", "gap", "k_k1_exchange")} for x in dev][:5]}


res = {"wrap512": {}, "depth8192": None, "ppl": {}}
for b in ("fp32kv", "fp32_bf16kv", "bf16kv"):
    res["wrap512"][b] = score(g / "wrap1" / f"wrap512_{b}.npz")
    print("wrap512", b, json.dumps(res["wrap512"][b])[:300], flush=True)
res["depth8192"] = score(g / "d8k" / "depth8192_fp32kv.npz")
print("depth8192", json.dumps(res["depth8192"])[:300], flush=True)
KVM = {"fp32kv": "fp32", "bf16kv": "bf16"}
for tag, evp in (("v1", "validation/megakernel_mid/qgate/eval_set_v1.json"),
                 ("full", "validation/megakernel_mid/qgate/eval_set_full_v1.json")):
    ev = json.loads(Path(evp).read_text())
    for b, kv in KVM.items():
        ref = R(kv).model
        rows = []
        with torch.inference_mode():
            for it in ev["ppl_items"]:
                t = it["tokens"]
                lg = ref.forward(torch.tensor([t]))[0].double()
                rows.append(torch.log_softmax(lg, -1)[:-1].gather(1, torch.tensor(t[1:]).view(-1, 1))[:, 0].numpy())
        rlp = np.asarray(rows)
        klp = np.load(g / f"ppl_{tag}" / f"ppl__{b}.npz")["logp"]
        rel = math.exp(float(-klp.mean())) / math.exp(float(-rlp.mean())) - 1
        res["ppl"][f"{tag}_{b}"] = {"ppl_rel_vs_ref": rel, "pass_1pct": abs(rel) < 1e-2,
                                    "max_abs_dlogp": float(np.abs(klp - rlp).max())}
        print("ppl", tag, b, f"{rel:+.3e}", flush=True)
out.write_text(json.dumps(res, indent=1))
