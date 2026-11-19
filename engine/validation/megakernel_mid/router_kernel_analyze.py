"""rec-tan.3 step 2: the kernel's own routing noise, from kernel-emitted step_experts.

For each npz from router_kernel_emit.py, the reference (fp32 compute, KV at the
build's tier) is teacher-forced on the kernel's tokens AND its routing
(step_experts) everywhere. At every MoE (layer, position) the kernel's expert set
is then compared with the reference's own top-k. Upstream routing is identical by
construction, so each deviation is a clean boundary event, not a downstream
consequence of another one. The kernel emits expert ids, not scores, so the
kernel's |dgap| itself cannot be measured. What is reported is the reference gap
at every boundary, and whether the kernel crossed it, binned by gap.

  python router_kernel_analyze.py --ckpt CK DIR --out kernel_calib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid_mk_check as H  # noqa: E402

BINS = [0.0, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, float("inf")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("kernel_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = H.vocab_of(a.ckpt)
    cases = H.case_matrix(vocab)
    refs: dict[str, H.Reference] = {}
    runs = []
    for f in sorted(a.kernel_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        kv, name = str(d["kv_dtype"]), str(d["case"])
        ref = refs.setdefault(kv, H.Reference(a.ckpt, device, kv))
        prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
        se, probs = H.step_experts_of(d, ref, len(prompt) + len(tokens))
        assert se is not None, (f.name, probs)
        dev, ref_f = H.routing_deviations(ref, prompt, tokens, se)
        _, gaps = ref.run(prompt, tokens, force=se)  # forced-trajectory gaps at every boundary
        ev_forced = H._evaluate(d, ref_f, kv, vocab)  # kernel vs reference with the kernel's routing
        ref_s, gaps_s = ref.run(prompt, tokens)
        ev = H._evaluate(d, ref_s, kv, vocab)
        router, _, ev_g, _, note = H.router_substitution_from_experts(ref, d, prompt, tokens, kv, vocab, ev, se)
        gate_pass = (not ev["problems"]) or router is not None
        if name in cases:  # the formal checker on the real cases, for cross-checking
            gate_pass_check = H.check(a.ckpt, f, device, verbose=False)["pass"]
        else:
            gate_pass_check = None
        g_all = np.array([g["gap"] for g in gaps])
        g_dev = np.array([x["gap"] for x in dev])
        runs.append({
            "file": f.name, "prompt": name, "tier": kv, "P": len(prompt),
            "boundaries": len(gaps), "deviations": dev, "n_dev": len(dev),
            "n_dev_not_k_k1": sum(not x["k_k1_exchange"] for x in dev),
            "hist_all": np.histogram(g_all, BINS)[0].tolist(),
            "hist_dev": np.histogram(g_dev, BINS)[0].tolist() if len(dev) else [0] * (len(BINS) - 1),
            "stock_pass": not ev["problems"], "stock_problems": ev["problems"],
            "gate_pass": gate_pass, "gate_pass_check": gate_pass_check, "router": router, "note": note,
            "forced_step_rel": ev_forced["step_rel"], "forced_last_rel": ev_forced["last_rel"],
            "forced_tokens_ok": not ev_forced["bad"], "stock_step_rel": ev["step_rel"],
        })
        r = runs[-1]
        print(f"{kv:4} {name:16} dev={r['n_dev']} (non-k/k+1 {r['n_dev_not_k_k1']}) "
              f"dev_gaps={[f'{x:.1e}' for x in sorted(g_dev)[:4]]} stock={'PASS' if r['stock_pass'] else 'FAIL'} "
              f"gate={'PASS' if gate_pass else 'FAIL'} forced step_rel {r['forced_step_rel']:.2e}"
              + (f" router L{router['layer']}@{router['pos']} gap {router['gap']:.2e}" if router else "")
              + (f" note: {note}" if note and not gate_pass else ""), flush=True)
        a.out.write_text(json.dumps({"ckpt_sha256": H.shard_sha256(a.ckpt), "bins": BINS[:-1] + ["inf"],
                                     "router_tie_tol": H.ROUTER_TIE_TOL, "runs": runs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
