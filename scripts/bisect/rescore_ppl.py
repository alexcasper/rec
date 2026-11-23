"""Dawn-side reproducible ppl scoring for rec-e5d rows: rescore committed kernel ppl npz (ppl__<build>.npz, emitted on dusk by
m3_depth_gate.py --ppl-set) against the torch fp32-compute CPU reference, with no dusk access needed.

Same math as qgate/score_ppl_full.py and scripts/bisect/score_e5d.py: reference = fp32 model, K/V stored at the build's KV tier
(Reference(kv) == fp32 model with kv_store_dtype set), teacher-forced full-sequence forward; ppl_rel = exp(nll_k)/exp(nll_ref) - 1.
Reference log-probs are cached per (ckpt shard sha, eval-set items sha, kv) under $REC_PPL_REF_CACHE.

  cd engine && PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python ../scripts/bisect/rescore_ppl.py \
      --manifest ../docs/benchmarks/e5d-ppl-manifest.json [--only SUBSTR] [--out report.json]
  ... rescore_ppl.py --dir DIR --ckpt mid|full|PATH --evalset eval_set_full_v1.json        (ad hoc)
ckpt classes resolve via $REC_CKPT_MID (default ~/rec-data/mid-prod) and $REC_CKPT_FULL (default ~/rec-data/flint-int4).
Exit 1 if any manifest row fails to reproduce its recorded ppl_rel within --tol (default 1e-9 absolute)."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk

ENGINE = Path(__file__).resolve().parents[2] / "engine"
QGATE = ENGINE / "validation/megakernel_mid/qgate"
ROOT = ENGINE.parent
CKPT = {"mid": Path(os.environ.get("REC_CKPT_MID", "~/rec-data/mid-prod")).expanduser(),
        "full": Path(os.environ.get("REC_CKPT_FULL", "~/rec-data/flint-int4")).expanduser()}
CACHE = Path(os.environ.get("REC_PPL_REF_CACHE", "~/rec-data/ppl_ref_cache")).expanduser()
KV = {"fp32kv": "fp32", "fp32_bf16kv": "bf16", "bf16kv": "bf16"}
LIM = {"fp32kv": 1e-4, "fp32_bf16kv": 1e-2, "bf16kv": 1e-2}
_models = {}


def ckpt_path(c):
    return CKPT[c] if c in CKPT else Path(c).expanduser()


_sha = {}


def ref_logp(ckpt: Path, ev: dict, kv: str) -> np.ndarray:
    if ckpt not in _sha:
        _sha[ckpt] = mk.shard_sha256(ckpt)   # hashes the whole first shard: once per process
    key = f"{_sha[ckpt][:16]}_{ev['sha256'][:16]}_{kv}.npy"
    f = CACHE / key
    if f.exists():
        return np.load(f)
    if ckpt not in _models:
        _models[ckpt] = mk.Reference(ckpt, "cpu", "fp32").model
    model = _models[ckpt]
    model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[kv]
    rows = []
    with torch.inference_mode():
        for it in ev["ppl_items"]:
            t = it["tokens"]
            lg = model.forward(torch.tensor([t]))[0].double()
            rows.append(torch.log_softmax(lg, -1)[:-1].gather(1, torch.tensor(t[1:]).view(-1, 1))[:, 0].numpy())
    out = np.asarray(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, out)
    return out


def score_dir(d: Path, ckpt: Path, evp: Path) -> dict:
    ev = json.loads(evp.read_text())
    res = {}
    for f in sorted(d.glob("ppl__*.npz")):
        b = f.stem.split("__")[1]
        z = np.load(f)
        assert str(z["evalset_sha256"]) == ev["sha256"], f"{f}: eval set mismatch"
        klp, rlp = z["logp"], ref_logp(ckpt, ev, KV[b])
        rel = math.exp(float(-klp.mean())) / math.exp(float(-rlp.mean())) - 1.0
        res[b] = {"ppl": math.exp(float(-klp.mean())), "ref_ppl": math.exp(float(-rlp.mean())), "ppl_rel_vs_ref": rel,
                  "limit": LIM[b], "pass": abs(rel) < LIM[b], "max_abs_dlogp": float(np.abs(klp - rlp).max())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--only", default="")
    ap.add_argument("--dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--evalset")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out")
    a = ap.parse_args()
    report, bad = [], 0
    if a.manifest:
        man = json.loads(Path(a.manifest).read_text())
        entries = [e for e in man["entries"] if a.only in e["id"]]
    else:
        entries = [{"id": a.dir, "dir": a.dir, "ckpt": a.ckpt, "evalset": a.evalset, "expected": {}}]
    for e in entries:
        d = Path(e["dir"]) if Path(e["dir"]).is_absolute() else ROOT / e["dir"]
        res = score_dir(d, ckpt_path(e["ckpt"]), QGATE / e["evalset"] if not Path(e["evalset"]).exists() else Path(e["evalset"]))
        for b, r in res.items():
            exp = e.get("expected", {}).get(b)
            r["expected"] = exp
            r["reproduces"] = None if exp is None else abs(r["ppl_rel_vs_ref"] - exp) <= a.tol
            bad += r["reproduces"] is False
            print(f"{e['id']:<44s} {b:<7s} rel {r['ppl_rel_vs_ref']:+.6e} {'PASS' if r['pass'] else 'FAIL'}"
                  + ("" if exp is None else f"  recorded {exp:+.6e} {'REPRODUCES' if r['reproduces'] else 'MISMATCH'}"),
                  flush=True)
        report.append({"id": e["id"], "builds": res})
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(report)} rows, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
