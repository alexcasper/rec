"""rec-3cr soak scoring (dawn): every DISTINCT full-config spot output vs one torch fp32 CPU reference (rec-pp2 fp32-KV tier).
Usage: PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python score_soak.py <soak_dir> <ckpt> <out.json>"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import mid_mk_check as mk

soak, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
files = sorted(soak.glob("*.npz"))
groups: dict[str, list[str]] = {}
for f in files:
    z = np.load(f)
    h = hashlib.md5()
    for k in ("tokens", "last_logits", "step_logits"):
        h.update(np.ascontiguousarray(z[k]).tobytes())
    groups.setdefault(h.hexdigest(), []).append(f.name)
print(f"{len(files)} npz -> {len(groups)} distinct outputs", flush=True)
vocab = mk.vocab_of(ckpt)
sha = mk.shard_sha256(ckpt)
refs: dict = {}
res = []
t0 = time.time()
for h, names in groups.items():
    d = np.load(soak / names[0])
    kv = str(d["kv_dtype"]) if "kv_dtype" in d.files else "fp32"
    if kv not in refs:
        refs[kv] = mk.Reference(ckpt, "cpu", kv)
    ref = refs[kv]
    assert str(d["ckpt_sha256"]) == sha, "ckpt sha mismatch"
    r = mk.check_arrays(ref, d, d["prompt"].tolist(), d["tokens"].tolist(), kv, vocab, str(d["case"]))
    r.pop("_ref", None)
    row = {"files": names, "kv": kv, "tier": mk.LOGIT_RTOL[kv], "tokens": d["tokens"].tolist(), "pass": r["pass"], "problems": r["problems"],
           "last_logits_rel": r["last_logits_rel"], "step_logits_max_rel": r["step_logits_max_rel"],
           "n_routing_deviations": r.get("n_routing_deviations"), "tokens_exact": r["tokens_exact"],
           "wall_s": round(time.time() - t0, 1)}
    res.append(row)
    print(json.dumps(row), flush=True)
out.write_text(json.dumps(res, indent=1))
