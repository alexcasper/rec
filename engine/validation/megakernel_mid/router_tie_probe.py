"""rec-tan adjudication probe: is the chat bf16-KV failure a router near-tie?

1. Run the rec-pp2 check on each kernel npz (unchanged gate).
2. Reference (fp32 compute, KV at the declared dtype, teacher-forced) with the
   MoE selection scores recorded. Report the smallest top-k vs (k+1) gaps.
3. Forced-routing replay: at the (layer, position) of the smallest gap, swap the
   k-th and (k+1)-th experts in the reference only, then recompare with the kernel.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid_mk_check as H  # noqa: E402

from rec_mlx.torch_rocm import load_model  # noqa: E402

CK = Path(sys.argv[1])
OUT = Path(sys.argv[2])
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(CK, device=dev, dtype=torch.float32)
K = model.cfg.topk

rec = []
orig_moe = type(model)._moe


def ref(prompt, tokens, kv, forced=None):
    """Teacher-forced reference; forced = (layer, pos, drop_expert, add_expert)."""
    model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[kv]
    rec.clear()
    real_topk = torch.topk

    def topk(t, k, dim=-1, **kw):
        r = real_topk(t, k, dim=dim, **kw)
        if forced is None or k != K or t.dim() != 3:
            return r
        L, p, drop, add = forced
        if topk.layer != L:
            return r
        idx = r.indices.clone()
        row = idx[0, p]
        assert drop in row.tolist(), (row.tolist(), drop)
        row[row == drop] = add
        return torch.return_types.topk((torch.gather(t, dim, idx), idx))

    def moe_wrap(self, x, i):
        topk.layer = i
        pre = f"model.layers.{i}.mlp"
        gate = torch.sigmoid(torch.nn.functional.linear(x.float(), self.t[f"{pre}.gate.weight"].float()))
        sel = gate + self.t[f"{pre}.gate.expert_bias"].float().view(1, 1, -1)
        srt = torch.sort(sel[0], dim=-1, descending=True)
        for p in range(sel.shape[1]):
            rec.append((i, p, float(srt.values[p, K - 1] - srt.values[p, K]),
                        int(srt.indices[p, K - 1]), int(srt.indices[p, K])))
        torch.topk = topk
        try:
            return orig_moe(self, x, i)
        finally:
            torch.topk = real_topk

    model._moe = moe_wrap.__get__(model)
    with torch.inference_mode():
        ids = torch.tensor([prompt + tokens], device=model.device)
        out = model.forward(ids)[0, len(prompt) - 1:]
    return out.float().cpu().numpy(), list(rec)


def compare(d, r):
    tokens = d["tokens"].astype(int)
    sl = d["step_logits"].astype(np.float64)
    r = r.astype(np.float64)
    step_rel = [float(np.abs(sl[i] - r[i]).max() / np.abs(r[i]).max()) for i in range(len(tokens))]
    last_rel = float(np.abs(d["last_logits"] - r[-1]).max() / np.abs(r[-1]).max())
    bad = []
    for i, t in enumerate(tokens):
        best = int(r[i].argmax())
        if t != best:
            bad.append({"step": i, "ref": best, "kernel": int(t),
                        "fp32_margin": float(r[i][best] - r[i][t])})
    return {"step_rel_max": max(step_rel), "worst_step": int(np.argmax(step_rel)),
            "last_rel": last_rel, "token_mismatches": bad}


res = {}
for f in sorted(OUT.glob("*.npz")):
    d = np.load(f, allow_pickle=False)
    case, kv = str(d["case"]), str(d["kv_dtype"])
    prompt = d["prompt"].astype(int).tolist()
    tokens = d["tokens"].astype(int).tolist()
    r, gaps = ref(prompt, tokens, kv)
    P0 = len(prompt) - 1
    dec = [g for g in gaps if g[1] >= P0]  # positions that produce a compared row
    dec.sort(key=lambda g: g[2])
    entry = {"case": case, "kv": kv, "stock": compare(d, r),
             "min_router_gaps_decode": [dict(layer=g[0], pos=g[1], step=g[1] - P0, gap=g[2],
                                             kth=g[3], next=g[4]) for g in dec[:4]]}
    if entry["stock"]["step_rel_max"] > H.LOGIT_RTOL[kv]:
        first_bad = next(i for i, v in enumerate(
            [float(np.abs(d["step_logits"][i] - r[i]).max() / np.abs(r[i]).max()) for i in range(len(tokens))])
            if v > H.LOGIT_RTOL[kv])
        # candidate ties at positions that can affect row first_bad (pos <= P0+first_bad)
        cands = sorted([g for g in gaps if g[1] <= P0 + first_bad], key=lambda g: g[2])[:3]
        entry["first_bad_step"] = first_bad
        entry["forced"] = []
        for g in cands:
            rf, _ = ref(prompt, tokens, kv, forced=(g[0], g[1], g[3], g[4]))
            entry["forced"].append({"layer": g[0], "pos": g[1], "step": g[1] - P0, "gap": g[2],
                                    "swap": [g[3], g[4]], "result": compare(d, rf)})
    res[f.name] = entry
    print(f.name, json.dumps(entry["stock"])[:300], flush=True)
json.dump(res, open(OUT / "router_tie_probe.json", "w"), indent=1)
print("WROTE", OUT / "router_tie_probe.json")
