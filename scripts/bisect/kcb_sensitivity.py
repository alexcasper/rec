"""rec-kcb root cause: is the full-config fp32_bf16kv strict-tier divergence (kernel vs torch bf16-KV ref: 2.74e-2, ~148 routing
devs on wrap512) real KV-tier drift, or bf16 rounding-boundary amplification of fp32 accumulation-order noise?

Kernel fp32_bf16kv == reference semantics (PURE_FP32 compute, RNE bf16 rounding of K/V after norm+RoPE, q unrounded), so the only
kernel-vs-ref difference is fp32 summation order upstream of the bf16 rounding. Emulate exactly that inside torch:
  P(eps): multiply every k_proj/v_proj output by (1 + eps*s), s = fixed random +-1 per element (eps ~ 1-2 fp32 ulp)
Teacher-forced on the kernel's wrap512 trajectory, UNFORCED routing, compare stock vs P(eps) reference:
  - bf16-KV:  if rel ~1e-2 and ~100+ expert-set differences appear from early positions -> inherent sensitivity (the strict tier
              cannot separate a correct kernel from the reference), not kernel drift
  - fp32-KV control: same perturbation must stay ~1e-6 (as the kernel's fp32kv tier does)
Plus kernel vs ref at bf16 KV: forced-routing rel and the gap distribution of the kernel's routing deviations.
  PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python kcb_sensitivity.py BF16_NPZ CKPT OUT.json [EPS ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mid_mk_check as mk

npz, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
EPS = [float(e) for e in sys.argv[4:]] or [2.0 ** -23, 2.0 ** -22]
d = np.load(npz)
prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
K = np.concatenate([d["step_logits"], d["last_logits"][None]], 0).astype(np.float64)


def rel(a, b):
    return [float(np.abs(a[i] - b[i]).max() / np.abs(b[i]).max()) for i in range(len(b))]


def setdiff(u1, u2):
    a, b = np.sort(u1, -1), np.sort(u2, -1)
    diff = np.argwhere((a != b).any(-1))           # (pos, layer)
    return {"n": int(len(diff)), "first_pos": int(diff[:, 0].min()) if len(diff) else None,
            "n_pos_lt_512": int((diff[:, 0] < 512).sum()) if len(diff) else 0,
            "layers_hist": np.bincount(diff[:, 1], minlength=u1.shape[1]).tolist() if len(diff) else []}


res = {"npz": str(npz), "P": len(prompt), "N": len(tokens), "eps": EPS, "runs": {}}
for kv in ("bf16", "fp32"):
    ref = mk.Reference(ckpt, "cpu", kv)
    model = ref.model
    base, _ = ref.run(prompt, tokens)
    base = base.astype(np.float64)
    used0 = ref.used.copy()
    orig_lin = model._lin
    for eps in EPS:
        gen = torch.Generator().manual_seed(20261123)
        signs = {}

        def lin(name, x, _eps=eps):
            y = orig_lin(name, x)
            if name.endswith(("k_proj", "v_proj")):
                key = (name, y.shape[-1])
                if key not in signs:
                    signs[key] = torch.randint(0, 2, (len(prompt) + len(tokens) + 1, y.shape[-1]), generator=gen).to(y.dtype) * 2 - 1
                s = signs[key][:y.shape[-2]] if y.dim() >= 2 else signs[key][0]
                y = y * (1 + _eps * s)
            return y

        model._lin = lin
        pert, _ = ref.run(prompt, tokens)
        model._lin = orig_lin
        r = rel(pert.astype(np.float64), base)
        sd = setdiff(ref.used, used0)
        res["runs"][f"{kv}_eps{eps:.3g}"] = {"rel_step0": r[0], "rel_max": max(r), "rel_last": r[-1],
                                             "expert_set_diffs": sd}
        print(f"[{kv} KV] stock vs perturbed eps={eps:.3g}: rel step0 {r[0]:.3e} max {max(r):.3e} | expert-set diffs "
              f"{sd['n']} (first pos {sd['first_pos']}, {sd['n_pos_lt_512']} in prompt)", flush=True)
    if kv == str(d["kv_dtype"]):
        rk = rel(K, base)
        sdk = setdiff(d["step_experts"][:len(used0)], used0)
        dev, ref_f = mk.routing_deviations(ref, prompt, tokens, d["step_experts"])
        rf = rel(K, ref_f.astype(np.float64))
        gaps = np.array([x["gap"] for x in dev]) if dev else np.zeros(0)
        res["kernel_vs_ref"] = {"unforced_rel_step0": rk[0], "unforced_rel_max": max(rk), "expert_set_diffs": sdk,
                                "forced_rel_step0": rf[0], "forced_rel_max": max(rf), "n_dev": len(dev),
                                "dev_gap_quantiles": np.quantile(gaps, [0, .25, .5, .75, .9, 1]).tolist() if len(gaps) else [],
                                "k_k1_exchange_frac": float(np.mean([x["k_k1_exchange"] for x in dev])) if dev else None,
                                "dev_pos_first": min(x["pos"] for x in dev) if dev else None}
        print(f"[{kv} KV] kernel vs ref: unforced step0 {rk[0]:.3e} max {max(rk):.3e} | forced step0 {rf[0]:.3e} max "
              f"{max(rf):.3e} | {len(dev)} devs, gap quantiles {res['kernel_vs_ref']['dev_gap_quantiles']}", flush=True)
    del ref, model
out.write_text(json.dumps(res, indent=1) + "\n")
