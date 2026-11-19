"""rec-tan.3: verify both step_experts policies (single = adopted, all-declared = proposed)
on the kernel calibration npz set (run from ~/rec-ve/pp2 on dusk, torch env)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "rec-ve/pp2/engine/validation/megakernel_mid"))
import mid_mk_check as H

CK = Path.home() / "rec-ve/ckpt-pp2/mid-prod"
vocab = H.vocab_of(CK)
refs = {}
rows = []
for f in sorted(Path("kcal/npz").glob("*.npz")):
    d = np.load(f)
    kv = str(d["kv_dtype"])
    ref = refs.setdefault(kv, H.Reference(CK, "cuda", kv))
    P, T = d["prompt"].tolist(), d["tokens"].tolist()
    se, _ = H.step_experts_of(d, ref, len(P) + len(T))
    r0, _ = ref.run(P, T)
    ev = H._evaluate(d, r0, kv, vocab)
    out = {"file": f.name}
    for pol in ("single", "all-declared"):
        H.ROUTER_EXPERTS_POLICY = pol
        rec, _, ev_s, dev, note = H.router_substitution_from_experts(ref, d, P, T, kv, vocab, ev, se)
        ok = (rec is not None) or (not ev["problems"] and not (ev_s and ev_s["problems"]))
        out[pol] = "PASS" if ok else "FAIL"
    if f.stem.split("_")[0] in H.case_matrix(vocab):
        for pol in ("single", "all-declared"):
            H.ROUTER_EXPERTS_POLICY = pol
            out["check_" + pol] = "PASS" if H.check(CK, f, "cuda", verbose=False)["pass"] else "FAIL"
    rows.append(out)
    print(json.dumps(out), flush=True)
json.dump(rows, open("kcal/policy_verify.json", "w"), indent=1)
print("DONE")
