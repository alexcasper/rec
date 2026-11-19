"""rec-tan.3: calibrate ROUTER_TIE_TOL (README 4a) on correct implementations.

For each prompt (the 3 rec-pp2 cases + seeded random prompts) and each KV tier:
  * stand-in kernel = the torch layer in kernel order (prefill, then one token per
    step, KV stored at the tier dtype), routing selection scores recorded per
    (layer, position);
  * reference = mid_mk_check.Reference (batch, teacher-forced on the stand-in's tokens).
Reports the streaming-vs-batch selection-score perturbation |d_sel|, top-k boundary
gaps, every routing flip (top-k set differs) with its gap, and whether the adopted
gate (stock, then router substitution) passes the run.

  python router_tie_calibrate.py --ckpt CK --seeds 10 --out calib.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid_mk_check as H  # noqa: E402


def prompts(vocab: int, seeds: int) -> dict[str, list[int]]:
    out = {name: spec["prompt"] for name, spec in H.case_matrix(vocab).items()}
    for s in range(1, seeds + 1):
        n = 512 if s % 2 == 0 else 32  # even seeds cross the 513 sliding window
        out[f"seed{s}-len{n}"] = [1] + [int(t) for t in np.random.default_rng(1000 + s).integers(6, vocab, n - 1)]
    return out


def stream(model, prompt, n_out):
    """Kernel-shaped decode; returns tokens, step logits, last logits, scores[layer] [P+n, E]."""
    import torch
    import torch.nn.functional as F

    rec: dict[int, list[np.ndarray]] = {}

    def moe(m, x, i):
        pre = f"model.layers.{i}.mlp"
        sel = torch.sigmoid(F.linear(x.float(), m.t[f"{pre}.gate.weight"].float())) \
            + m.t[f"{pre}.gate.expert_bias"].float().view(1, 1, -1)
        rec.setdefault(i, []).append(sel[0].cpu().numpy())
        return type(m)._moe(m, x, i)

    model._moe = types.MethodType(moe, model)
    try:
        with torch.inference_mode():
            kv = model.new_cache()
            logits = model.forward(torch.tensor([prompt], device=model.device), kv)[0, -1]
            tokens, steps = [], []
            for _ in range(n_out):
                steps.append(logits.float().cpu().numpy())
                tokens.append(int(logits.argmax()))
                logits = model.forward(torch.tensor([[tokens[-1]]], device=model.device), kv)[0, -1]
    finally:
        del model._moe
    return tokens, np.stack(steps), logits.float().cpu().numpy(), {i: np.concatenate(v) for i, v in rec.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiers", default="bf16,fp32")
    a = ap.parse_args()
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = H.vocab_of(a.ckpt)
    ps = prompts(vocab, a.seeds)
    ref = H.Reference(a.ckpt, device, "fp32")  # one model; tier set via kv_store_dtype
    K = ref.k
    runs = []
    for tier in a.tiers.split(","):
        ref.model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[tier]
        for name, prompt in ps.items():
            t0 = time.time()
            tokens, steps, last, s_scores = stream(ref.model, prompt, H.N_OUT)
            r_logits, gaps = ref.run(prompt, tokens)
            r_scores = ref.scores
            gap_at = {(g["layer"], g["pos"]): g["gap"] for g in gaps}
            # dsel: max over experts of |stream - ref| selection score.
            # dgap: how far the stand-in moves the reference's rank-k vs rank-(k+1)
            # boundary, (r_k - r_k1) - (s_k - s_k1). A flip there needs dgap > gap,
            # so max|dgap| is the quantity ROUTER_TIE_TOL must cover.
            dsel, dgap, flips, near = [], [], [], []
            for layer, rs in r_scores.items():
                ss = s_scores[layer]
                assert ss.shape == rs.shape, (ss.shape, rs.shape)
                d = np.abs(ss - rs).max(-1)
                dsel.extend(d.tolist())
                order = np.argsort(-rs, axis=-1)
                for p in range(rs.shape[0]):
                    gap = gap_at[(layer, p)]
                    k_, k1 = order[p, K - 1], order[p, K]
                    dg = float((rs[p, k_] - rs[p, k1]) - (ss[p, k_] - ss[p, k1]))
                    dgap.append(abs(dg))
                    rec = {"layer": layer, "pos": p, "gap": gap, "dgap": dg, "dsel": float(d[p])}
                    if set(np.argsort(-ss[p])[:K]) != set(order[p, :K]):
                        flips.append(rec)
                    elif gap <= 1e-3:
                        near.append(rec)
            kd = {"tokens": np.array(tokens), "last_logits": last, "step_logits": steps}
            ev = H._evaluate(kd, r_logits, tier, vocab)
            router, tried, _, ev_s = H.router_substitution(ref, kd, prompt, tokens, tier, vocab, ev, gaps)
            ref.model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[tier]  # run() keeps it; be explicit
            runs.append({
                "tier": tier, "prompt": name, "P": len(prompt),
                "stock_pass": not ev["problems"], "stock_problems": ev["problems"],
                "gate_pass": (not ev["problems"]) or router is not None, "router": router, "tried": tried,
                "step_rel": ev["step_rel"], "last_rel": ev["last_rel"],
                "dsel_max": float(max(dsel)), "dsel_p99": float(np.quantile(dsel, 0.99)),
                "dsel_median": float(np.median(dsel)), "min_gap": float(min(gap_at.values())),
                "dgap_max": float(max(dgap)), "dgap_p99": float(np.quantile(dgap, 0.99)),
                "dgap_p999": float(np.quantile(dgap, 0.999)), "n_boundaries": len(dgap),
                "dgap_all": [round(x, 9) for x in dgap],
                "flips": flips, "near_boundaries": sorted(near, key=lambda x: x["gap"])[:10],
                "seconds": round(time.time() - t0, 1),
            })
            r = runs[-1]
            print(f"{tier:4} {name:16} stock={'PASS' if r['stock_pass'] else 'FAIL'} "
                  f"gate={'PASS' if r['gate_pass'] else 'FAIL'} flips={len(flips)} "
                  f"dgap max {r['dgap_max']:.2e} p99 {r['dgap_p99']:.2e} dsel max {r['dsel_max']:.2e} "
                  f"min_gap {r['min_gap']:.2e} "
                  f"step_rel {r['step_rel']:.2e} {r['seconds']}s"
                  + (f" router L{router['layer']}@{router['pos']} gap {router['gap']:.2e}" if router else ""),
                  flush=True)
            a.out.write_text(json.dumps({"ckpt_sha256": H.shard_sha256(a.ckpt), "topk": K,
                                         "router_tie_tol": H.ROUTER_TIE_TOL, "runs": runs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
