"""rec-21a CPU Go/No-Go: W4A8 = int8 activations x the existing INT4-affine g64 weights, teacher-forced on eval_set_full_v1
(torch fp32 CPU reference, full flint-int4). Activation fake-quant (symmetric int8, RNE, dequant before the matmul) is applied to
the input of EVERY INT4-weight linear the megakernel runs as a q4 gemv: attention q/k/v/o, routed experts gate/up/down, shared
experts, dense MLPs, lm_head. The fp32 MoE router (F.linear with the [num_experts, hidden] gate weight) is left unquantized, as in
the kernel. KV store stays fp32 (separate axis, rec-ywy).
Variants: a8_tok  per-token absmax over the full K (what decode sees: L = 1)
          a8_g64  per-token, per 64-element group (aligned with the INT4 weight groups; the kernel needs per-group partials anyway)
          a8_static per-linear-name per-tensor static scale, calibrated on eval_set_v1 (clips beyond the calibrated range)
          a8_tok_nolm  as a8_tok but lm_head input kept fp32
Gate (rec-21a): ppl |delta| < 1% AND top-1 >= 97% vs the fp32-activation reference.
    PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python w4a8_ladder.py CKPT EVAL_FULL EVAL_CALIB OUT.json [variant ...]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_real

import mid_mk_check as mk
import rec_mlx.torch_rocm as tr

ckpt, evp, calp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ONLY = set(sys.argv[5:])
ev, cal = json.loads(evp.read_text()), json.loads(calp.read_text())
ref = mk.Reference(ckpt, "cpu", "fp32")
model = ref.model
model.kv_store_dtype = None
cfg = model.cfg
NE, HID = cfg.num_experts, cfg.hidden
_c = json.loads((ckpt / "config.json").read_text())
_hd, _nh, _nkv = _c.get("head_dim", 128), _c["num_attention_heads"], _c["num_key_value_heads"]
ATTN_SHAPES = {(_nh * _hd, HID), (_nkv * _hd, HID), (HID, _nh * _hd)}
MOE_SHAPES = {(_c["moe_intermediate_size"], HID), (HID, _c["moe_intermediate_size"])}


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def q_tok(x, G=None):
    if G is None:
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
        return torch.clamp(torch.round(x / s), -127, 127) * s
    sh = x.shape
    g = x.reshape(*sh[:-1], sh[-1] // G, G)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
    return (torch.clamp(torch.round(g / s), -127, 127) * s).reshape(sh)


class FProxy:
    """Stands in for torch.nn.functional inside rec_mlx.torch_rocm: quantizes linear inputs except the router."""
    def __init__(self):
        self.mode, self.static, self.calib, self.skip_lm, self.n_q, self.clip = None, {}, None, False, 0, [0, 0]

    def __getattr__(self, k):
        return getattr(F_real, k)

    def key(self, w):
        return tuple(w.shape)

    def linear(self, x, w, b=None):
        router = w.dim() == 2 and w.shape[0] == NE and w.shape[1] == HID
        lm = w.shape[0] == cfg.vocab
        if self.calib is not None and not router:
            k = self.key(w)
            self.calib[k] = max(self.calib.get(k, 0.0), float(x.abs().max()))
        if self.mode and not router and not (lm and self.skip_lm):
            self.n_q += 1
            if self.mode == "tok":
                x = q_tok(x)
            elif self.mode == "g64":
                x = q_tok(x, 64)
            elif self.mode in ("g32", "g16"):
                x = q_tok(x, int(self.mode[1:]))
            elif self.mode.startswith("g64:"):          # class ablation: quantize only one linear class (g64)
                cls = self.mode[4:]
                kind = ("lm" if lm else "attn" if w.shape in ATTN_SHAPES else "moe" if w.shape in MOE_SHAPES else "shared")
                if kind == cls:
                    x = q_tok(x, 64)
            elif self.mode == "static":
                s = self.static[self.key(w)] / 127.0
                self.clip[0] += int((x.abs() > self.static[self.key(w)]).sum()); self.clip[1] += x.numel()
                x = torch.clamp(torch.round(x / s), -127, 127) * s
        return F_real.linear(x, w, b)


FP = FProxy()
tr.F = FP          # torch_rocm uses `F.linear` for every projection, expert, shared expert, lm_head and the router


def run():
    logits, ex = [], []
    with torch.inference_mode():
        for it in ev["ppl_items"]:
            logits.append(model.forward(torch.tensor([it["tokens"]]))[0].float().numpy())
    return logits


def metrics(a, b):
    rels, flips, agree, lpa, lpb = [], [], [], [], []
    for la, lb, it in zip(a, b, ev["ppl_items"]):
        la64, lb64 = la.astype(np.float64), lb.astype(np.float64)
        rels.append(np.abs(la64 - lb64).max(-1) / np.abs(lb64).max(-1))
        am, bm = la64[:-1].argmax(-1), lb64[:-1].argmax(-1)
        d = np.nonzero(am != bm)[0]
        flips.append(int(d[0]) if len(d) else None)
        agree.append(float((am == bm).mean()))
        t = np.asarray(it["tokens"][1:])
        for lg, acc in ((la64[:-1], lpa), (lb64[:-1], lpb)):
            m = lg.max(-1, keepdims=True)
            acc.append((lg - m - np.log(np.exp(lg - m).sum(-1, keepdims=True)))[np.arange(len(t)), t])
    R = np.concatenate(rels)
    pa, pb = math.exp(-np.concatenate(lpa).mean()), math.exp(-np.concatenate(lpb).mean())
    fl = [f for f in flips if f is not None]
    r = {"rel_max": float(R.max()), "rel_pos0": float(max(x[0] for x in rels)), "rel_median": float(np.median(R)),
         "first_flip_pos_min": min(fl) if fl else None, "items_with_flip": len(fl), "top1_agree": float(np.mean(agree)),
         "ppl": pa, "ppl_rel": pa / pb - 1, "max_abs_dlogp": float(np.abs(np.concatenate(lpa) - np.concatenate(lpb)).max())}
    r["gate_pass"] = abs(r["ppl_rel"]) < 0.01 and r["top1_agree"] >= 0.97
    return r


res = json.loads(out.read_text()) if out.exists() else {}
res.setdefault("meta", {"evalset_sha256": ev["sha256"], "calib_sha256": cal.get("sha256"), "threads": torch.get_num_threads()})
res.setdefault("vs_fp32_act", {})
t0 = time.time()
base = run()
log(f"fp32-activation base done ({time.time() - t0:.0f}s)")
VARIANTS = [("a8_tok", "tok", False), ("a8_g64", "g64", False), ("a8_static", "static", False), ("a8_tok_nolm", "tok", True),
            ("a8_g32", "g32", False), ("a8_g16", "g16", False), ("a8_g64_attn_only", "g64:attn", False),
            ("a8_g64_moe_only", "g64:moe", False), ("a8_g64_shared_only", "g64:shared", False), ("a8_g64_lm_only", "g64:lm", False)]
for name, mode, skip_lm in VARIANTS:
    if ONLY and name not in ONLY:
        continue
    if mode == "static" and not FP.static:
        FP.calib = {}
        with torch.inference_mode():
            for s in [it["tokens"] for it in cal["ppl_items"]] + [it["prompt"] for it in cal["gen_items"]]:
                model.forward(torch.tensor([s]))
        FP.static, FP.calib = FP.calib, None
        log(f"static scales calibrated on eval_set_v1 for {len(FP.static)} weight shapes")
    FP.mode, FP.skip_lm, FP.n_q, FP.clip = mode, skip_lm, 0, [0, 0]
    t0 = time.time()
    r = run()
    FP.mode = None
    m = metrics(r, base)
    m["quantized_linear_calls"] = FP.n_q
    if mode == "static":
        m["clip_frac"] = FP.clip[0] / max(FP.clip[1], 1)
    res["vs_fp32_act"][name] = m
    log(f"{name} ({time.time() - t0:.0f}s): " + json.dumps({k: m[k] for k in ("ppl_rel", "top1_agree", "rel_pos0", "first_flip_pos_min", "gate_pass")}))
    out.write_text(json.dumps(res, indent=1) + "\n")
log("done")
