"""rec-hyf: KV-precision ladder + bf16-KV fix candidates, in the rec-kcb teacher-forced harness (torch fp32 CPU reference,
eval_set_full_v1 8x192 real text, perturbation P(eps) from kcb_sensitivity.py / kcb_ppl_floor.py).

Every scheme is a fake-quant applied at the KV store point (after k_norm + RoPE, the same point as kv_store_dtype),
dequant-on-read; compute stays fp32. Metrics per scheme vs the fp32-KV stock run, from position 0:
  max-rel logit divergence (max|dl| / max|l_ref| per position; max over all positions, at pos 0, median),
  first argmax-flip position (per item), top-1 agreement, ppl rel, MoE expert-set differences (pos, layer).
Amplification test (eb06650 disease): scheme(stock) vs scheme(P(eps), eps = 1 fp32 ulp) -> same metrics.

Schemes
  fp32, bf16 (RNE, == kv_store_dtype=bf16)
  int{8,4}_pc   symmetric per-channel STATIC scales (absmax per layer x kv-head x dim, calibrated on eval_set_v1 under fp32
                KV; values beyond the calibrated range clip), int{8,4}_pt per-token-per-head dynamic absmax over head_dim,
                int{8,4}_g32 per-token group-32 dynamic (llama.cpp q8_0 / q4_0-style symmetric blocks)
  bf16_sr       stochastic rounding to bf16 (unbiased), seeded generator; amplification run reuses the same seed
  bf16_scaled   per-channel static scale before bf16 RNE (x/s rounded, *s on read)
  bf16_router   ORACLE: bf16 RNE KV, MoE top-k replaced by the fp32-KV stock run's experts (the router reads the residual
                stream, not KV; this bounds what decoupling routing decisions from bf16 noise can buy)
    PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python kv_ladder.py CKPT EVAL_FULL EVAL_CALIB OUT.json [run ...]
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import mid_mk_check as mk

ckpt, evp, calp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ONLY = set(sys.argv[5:])
EPS = 2.0 ** -23
ev = json.loads(evp.read_text())
cal = json.loads(calp.read_text())
ref = mk.Reference(ckpt, "cpu", "fp32")
model = ref.model
model.kv_store_dtype = None
orig_lin = model._lin
E = model.cfg.experts if hasattr(model.cfg, "experts") else None
TOPK = model.cfg.topk

# ---------------------------------------------------------------- top-k record / replay (MoE router only)
_orig_topk = torch.topk
TK = SimpleNamespace(mode=None, rec=[], replay=None, i=0)


def topk_hook(x, k, dim=-1, *a, **kw):
    r = _orig_topk(x, k, dim, *a, **kw)
    if k != TOPK or TK.mode is None:
        return r
    if TK.mode == "record":
        TK.rec.append(r.indices.clone())
        return r
    idx = TK.replay[TK.i]
    TK.i += 1
    TK.rec.append(idx)
    return SimpleNamespace(indices=idx, values=torch.gather(x, dim, idx))


torch.topk = topk_hook


# ---------------------------------------------------------------- quantizers ([B, kvh, L, D])
def q_sym(x, s, Q):
    return torch.clamp(torch.round(x / s), -Q, Q) * s


def mk_pc(Q, scales, stats):
    def f(i, k, v):
        sk, sv = scales[i]
        for t, s, nm in ((k, sk, "k"), (v, sv, "v")):
            stats[nm + "_clip"] += int((t.abs() > s.view(1, -1, 1, t.shape[-1]) * 1.0).sum())
            stats[nm + "_n"] += t.numel()
        return (q_sym(k, sk.view(1, -1, 1, k.shape[-1]) / Q, Q), q_sym(v, sv.view(1, -1, 1, v.shape[-1]) / Q, Q))
    return f


def mk_tok(Q, G):
    def one(t):
        B, H, L, D = t.shape
        g = t.reshape(B, H, L, D // G, G)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / Q
        return q_sym(g, s, Q).reshape(B, H, L, D)
    return lambda i, k, v: (one(k), one(v))


def mk_pc_seq(Q):
    """per-channel scales from the sequence's own absmax (no clipping): upper bound for any well-calibrated static scale."""
    def one(t):
        return q_sym(t, t.abs().amax(2, keepdim=True).clamp_min(1e-30) / Q, Q)
    return lambda i, k, v: (one(k), one(v))


def mk_kivi(Q, G=128):
    """KIVI-style: K per-channel over G-token groups (causal per group), V per-token over head_dim."""
    def kq(t):
        B, H, L, D = t.shape
        out = torch.empty_like(t)
        for a in range(0, L, G):
            g = t[:, :, a:a + G]
            out[:, :, a:a + G] = q_sym(g, g.abs().amax(2, keepdim=True).clamp_min(1e-30) / Q, Q)
        return out
    vq = mk_tok(Q, 128)
    return lambda i, k, v: (kq(k), vq(i, k, v)[1])


def bf16_rne(t):
    return t.to(torch.bfloat16).to(torch.float32)


def mk_sr(seed):
    gen = torch.Generator().manual_seed(seed)

    def one(t):
        b = t.contiguous().view(torch.int32)
        r = torch.randint(0, 1 << 16, b.shape, generator=gen, dtype=torch.int32)
        return ((b + r) & ~0xFFFF).view(torch.float32)   # finite inputs only (KV after norm: |x| << max)
    return lambda i, k, v: (one(k), one(v))


def mk_scaled(scales):
    def f(i, k, v):
        sk, sv = scales[i]
        sk, sv = sk.view(1, -1, 1, k.shape[-1]), sv.view(1, -1, 1, v.shape[-1])
        return bf16_rne(k / sk) * sk, bf16_rne(v / sv) * sv
    return f


# ---------------------------------------------------------------- runs
def perturb(seed):
    gen = torch.Generator().manual_seed(seed)
    signs = {}

    def lin(name, x):
        y = orig_lin(name, x)
        if name.endswith(("k_proj", "v_proj")):
            key = (name, y.shape[-1])
            if key not in signs:
                signs[key] = torch.randint(0, 2, (y.shape[-2], y.shape[-1]), generator=gen).to(y.dtype) * 2 - 1
            y = y * (1 + EPS * signs[key])
        return y
    return lin


def run(quant, pert_seed=None, topk_mode=None, replay=None):
    model.kv_quant = quant
    model._lin = perturb(pert_seed) if pert_seed else orig_lin
    TK.mode, TK.rec, TK.replay, TK.i = topk_mode or "record", [], replay, 0
    logits = []
    with torch.inference_mode():
        for it in ev["ppl_items"]:
            logits.append(model.forward(torch.tensor([it["tokens"]]))[0].float().numpy())
    model.kv_quant, model._lin, TK.mode = None, orig_lin, None
    L = logits[0].shape[0]
    # router choices: one [1, L, k] tensor per (item, layer) in call order -> [items, L, layers, k]
    rec = TK.rec
    nl = len(rec) // len(logits)
    ex = np.stack([np.stack([rec[j * nl + l][0].numpy() for l in range(nl)], 1) for j in range(len(logits))])
    return {"logits": logits, "experts": ex, "replay": rec}


def metrics(a, b):
    """a vs b (b = baseline)."""
    rels, flips, agree, lpa, lpb = [], [], [], [], []
    for la, lb, it in zip(a["logits"], b["logits"], ev["ppl_items"]):
        la64, lb64 = la.astype(np.float64), lb.astype(np.float64)
        r = np.abs(la64 - lb64).max(-1) / np.abs(lb64).max(-1)
        rels.append(r)
        am, bm = la64.argmax(-1), lb64.argmax(-1)
        d = np.nonzero(am != bm)[0]
        flips.append(int(d[0]) if len(d) else None)
        agree.append(float((am == bm).mean()))
        t = np.asarray(it["tokens"][1:])
        for lg, acc in ((la64, lpa), (lb64, lpb)):
            m = lg.max(-1, keepdims=True)
            ls = lg - m - np.log(np.exp(lg - m).sum(-1, keepdims=True))
            acc.append(ls[np.arange(len(t)), t])
    R = np.concatenate(rels)
    ppa, ppb = math.exp(-np.concatenate(lpa).mean()), math.exp(-np.concatenate(lpb).mean())
    sa, sb = np.sort(a["experts"], -1), np.sort(b["experts"], -1)
    diff = np.argwhere((sa != sb).any(-1))  # (item, pos, layer)
    fl = [f for f in flips if f is not None]
    return {"rel_max": float(R.max()), "rel_pos0": float(max(r[0] for r in rels)), "rel_median": float(np.median(R)),
            "first_flip_pos_min": min(fl) if fl else None, "first_flip_pos_per_item": flips,
            "items_with_flip": len(fl), "top1_agree": float(np.mean(agree)),
            "ppl": ppa, "ppl_rel": ppa / ppb - 1,
            "max_abs_dlogp": float(np.abs(np.concatenate(lpa) - np.concatenate(lpb)).max()),
            "expert_set_diffs": int(len(diff)), "expert_diff_first_pos": int(diff[:, 1].min()) if len(diff) else None}


res = json.loads(out.read_text()) if out.exists() else {}
res.setdefault("meta", {"evalset_sha256": ev["sha256"], "calib": str(calp), "calib_sha256": cal.get("sha256"), "eps": EPS,
                        "threads": torch.get_num_threads()})
res.setdefault("vs_fp32", {}), res.setdefault("amplification", {})


def save():
    out.write_text(json.dumps(res, indent=1) + "\n")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# calibration: per-layer absmax over eval_set_v1 (ppl + gen prompts), fp32 KV
amax = {}


def calib_hook(i, k, v):
    ka, va = k.abs().amax((0, 2)), v.abs().amax((0, 2))
    if i in amax:
        amax[i] = (torch.maximum(amax[i][0], ka), torch.maximum(amax[i][1], va))
    else:
        amax[i] = (ka, va)
    return k, v


t0 = time.time()
amax_cache = out.with_suffix(".calib.pt")
if amax_cache.exists():
    amax = torch.load(amax_cache)
    log(f"calibration loaded from {amax_cache}")
elif ONLY and not any(n.endswith(("_pc", "scaled")) or n == "all" for n in ONLY):
    log("calibration skipped (no static-scale scheme selected)")
else:
    model.kv_quant = calib_hook
    with torch.inference_mode():
        seqs = [it["tokens"] for it in cal["ppl_items"]] + [it["prompt"] for it in cal["gen_items"] if "prompt" in it]
        for s in seqs:
            model.forward(torch.tensor([s]))
    model.kv_quant = None
    torch.save(amax, amax_cache)
    log(f"calibrated on {len(seqs)} eval_set_v1 sequences ({sum(map(len, seqs))} tokens) in {time.time() - t0:.0f}s")

base = run(None)
log("fp32 stock done")
rep = run(None)
res["harness_repeat_fp32"] = metrics(rep, base)
log(f"fp32 repeat vs fp32: rel_max {res['harness_repeat_fp32']['rel_max']:.3e} (harness noise)")
save()

RUNS = []
for Q, nm in ((127, "int8"), (7, "int4")):
    RUNS += [(f"{nm}_pc", lambda Q=Q: mk_pc(Q, amax, STATS.setdefault(f"q{Q}", {"k_clip": 0, "k_n": 0, "v_clip": 0, "v_n": 0}))),
             (f"{nm}_pt", lambda Q=Q: mk_tok(Q, 128)), (f"{nm}_g32", lambda Q=Q: mk_tok(Q, 32)),
             (f"{nm}_pcseq", lambda Q=Q: mk_pc_seq(Q)), (f"{nm}_kivi", lambda Q=Q: mk_kivi(Q))]
STATS = {}
BF = [("bf16", lambda: (lambda i, k, v: (bf16_rne(k), bf16_rne(v)))),
      ("bf16_sr_s1", lambda: mk_sr(1)), ("bf16_sr_s2", lambda: mk_sr(2)),
      ("bf16_sr_unseeded", lambda: mk_sr(int.from_bytes(os.urandom(4), "little"))),
      ("bf16_scaled", lambda: mk_scaled(amax))]
AMP = {"fp32": None, "bf16": None, "int8_pc": None, "int8_g32": None, "int8_kivi": None, "bf16_sr_s1": None, "bf16_scaled": None}

# 1) amplification control: fp32 under P(eps)
for seed in (1, 2):
    k = f"fp32_eps_s{seed}"
    if not ONLY or "amp" in ONLY:
        res["amplification"][k] = metrics(run(None, seed), base)
        log(f"AMP {k}: {json.dumps({x: res['amplification'][k][x] for x in ('rel_max', 'ppl_rel', 'expert_set_diffs')})}")
        save()

for name, mkq in BF + RUNS:
    if ONLY and name not in ONLY and "all" not in ONLY:
        continue
    t0 = time.time()
    r = run(mkq())
    res["vs_fp32"][name] = metrics(r, base)
    log(f"{name} vs fp32 ({time.time() - t0:.0f}s): " + json.dumps({x: res['vs_fp32'][name][x] for x in
        ('rel_max', 'rel_pos0', 'first_flip_pos_min', 'top1_agree', 'ppl_rel', 'expert_set_diffs')}))
    if name in AMP:
        seeds = (1, 2) if name == "bf16" else (1,)
        for seed in seeds:
            p = run(mkq(), seed)            # same quantizer construction -> same SR draws for the seeded SR run
            res["amplification"][f"{name}_eps_s{seed}"] = metrics(p, r)
            log(f"AMP {name}_eps_s{seed}: " + json.dumps({x: res['amplification'][f'{name}_eps_s{seed}'][x] for x in
                ('rel_max', 'rel_pos0', 'first_flip_pos_min', 'ppl_rel', 'expert_set_diffs')}))
    if STATS:
        res["clip_stats"] = STATS
    save()

# 2) router-decoupling oracle: bf16 RNE KV, routing replayed from fp32 stock (and its amplification test)
bfq = lambda i, k, v: (bf16_rne(k), bf16_rne(v))  # noqa: E731
if not ONLY or "router" in ONLY or "all" in ONLY:
    r = run(bfq, None, "replay", base["replay"])
    res["vs_fp32"]["bf16_router_oracle"] = metrics(r, base)
    log("bf16_router_oracle vs fp32: " + json.dumps({x: res['vs_fp32']['bf16_router_oracle'][x] for x in
        ('rel_max', 'rel_pos0', 'first_flip_pos_min', 'top1_agree', 'ppl_rel', 'expert_set_diffs')}))
    p = run(bfq, 1, "replay", base["replay"])
    res["amplification"]["bf16_router_oracle_eps_s1"] = metrics(p, r)
    log("AMP bf16_router_oracle_eps_s1: " + json.dumps({x: res['amplification']['bf16_router_oracle_eps_s1'][x] for x in
        ('rel_max', 'rel_pos0', 'ppl_rel', 'expert_set_diffs')}))
    save()
log("done")
