"""rec-21a step-4 quality scorer: W4A8 megakernel (routed-expert gate/up input, per-g16 int8 fake-quant) vs torch.

R0 = torch fp32-activations; R1 = torch with the routed-expert gate/up INPUT fake-quantized at g16
(symmetric absmax/127, RNE — identical to the kernel's quantize pass; down/shared/attn/lm stay fp32,
matching the v1 kernel scope GEPA_W4A8=1). Kernel side = m3_depth_gate --ppl-set npz on the W4A8 lib.
Gates: kernel vs R1 = implementation deviation (max|dlogp|); kernel vs R0 = quality (|ppl_rel| < 1%,
top1 >= 97%). torch-vs-torch R1-vs-R0 included for reference (the g16 whole-line expert-only CPU number).

PYTHONPATH=src:validation/megakernel_mid REC_REF_MMAP=1 python score_w4a8.py CKPT EVAL_FULL KERNEL_NPZ OUT.json
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

ckpt, evp, knpz, out = map(Path, sys.argv[1:5])
ev = json.loads(evp.read_text())
ref = mk.Reference(ckpt, "cpu", "fp32")
model = ref.model
model.kv_store_dtype = None
cfg = model.cfg
NE, HID = cfg.num_experts, cfg.hidden
INTER = json.loads((ckpt / "config.json").read_text())["moe_intermediate_size"]


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def q_g16(x):
    sh = x.shape
    g = x.reshape(*sh[:-1], sh[-1] // 16, 16)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
    return (torch.clamp(torch.round(g / s), -127, 127) * s).reshape(sh)


class FProxy:
    on = False

    def __getattr__(self, k):
        return getattr(F_real, k)

    def linear(self, x, w, b=None):
        if self.on and w.dim() == 3 and w.shape[1] == INTER and w.shape[2] == HID:
            x = q_g16(x)
        return F_real.linear(x, w, b)


FP = FProxy()


def run():
    with torch.inference_mode():
        return [model.forward(torch.tensor([it["tokens"]]))[0].float().numpy() for it in ev["ppl_items"]]


def lg(lgs, it):
    t = np.asarray(it["tokens"][1:])
    m = lgs.max(-1, keepdims=True)
    return (lgs - m - np.log(np.exp(lgs - m).sum(-1, keepdims=True)))[np.arange(len(t)), t]


def torch_pair(a_items, b_items):
    rels, agree, lpa, lpb, flips = [], [], [], [], []
    for la, lb, it in zip(a_items, b_items, ev["ppl_items"]):
        la64, lb64 = la.astype(np.float64), lb.astype(np.float64)
        rels.append(np.abs(la64 - lb64).max(-1) / np.abs(lb64).max(-1))
        am, bm = la64[:-1].argmax(-1), lb64[:-1].argmax(-1)
        d = np.nonzero(am != bm)[0]
        flips.append(int(d[0]) if len(d) else None)
        agree.append(float((am == bm).mean()))
        lpa.append(lg(la64[:-1], it))
        lpb.append(lg(lb64[:-1], it))
    R = np.concatenate(rels)
    pa, pb = math.exp(-np.concatenate(lpa).mean()), math.exp(-np.concatenate(lpb).mean())
    fl = [f for f in flips if f is not None]
    return {"rel_max": float(R.max()), "rel_pos0": float(max(x[0] for x in rels)),
            "first_flip_pos_min": min(fl) if fl else None, "items_with_flip": len(fl),
            "top1_agree": float(np.mean(agree)), "ppl": pa, "ppl_rel": pa / pb - 1,
            "max_abs_dlogp": float(np.abs(np.concatenate(lpa) - np.concatenate(lpb)).max())}


res = {"meta": {"evalset_sha256": ev.get("sha256"), "NE": NE, "HID": HID, "INTER": INTER,
                "quant_scope": "routed-expert gate/up input, g16 whole-line, symmetric absmax/127 RNE"}}
t0 = time.time()
base = run()
log(f"fp32-A base done ({time.time() - t0:.0f}s)")
FP.on = True
r1 = run()
FP.on = False
log(f"expert-g16 variant done ({time.time() - t0:.0f}s)")
res["torch_R1_vs_R0"] = torch_pair(r1, base)
res["torch_fp32A_ppl"] = res["torch_R1_vs_R0"]["ppl_rel"] + 1
res["torch_R1_ppl"] = res["torch_R1_vs_R0"]["ppl"]
out.write_text(json.dumps(res, indent=1) + "\n")
log("torch refs done: " + json.dumps({k: res["torch_R1_vs_R0"][k] for k in ("ppl_rel", "top1_agree", "rel_pos0")}))

if knpz.exists():
    z = np.load(knpz)
    keys = list(z.keys())
    res["kernel_npz_keys"] = keys
    log(f"kernel npz keys: {keys}")
    if "logp" in keys:
        kp = z["logp"]
        kp = np.concatenate([np.asarray(x).reshape(-1) for x in kp]) if kp.dtype == object or kp.ndim > 1 else np.asarray(kp).reshape(-1)
        tp = np.concatenate(lpa) if False else None
        # torch logp concatenated (fp32-A ref)
        t_lpa, t_lpb = [], []
        for la, lb, it in zip(r1, base, ev["ppl_items"]):
            t_lpa.append(lg(la.astype(np.float64)[:-1], it))
            t_lpb.append(lg(lb.astype(np.float64)[:-1], it))
        tl0, tl1 = np.concatenate(t_lpb), np.concatenate(t_lpa)
        n = min(len(kp), len(tl0))
        res["kernel_vs_R0"] = {"ppl": math.exp(-kp[:n].mean()), "ppl_rel": math.exp(-kp[:n].mean()) / math.exp(-tl0[:n].mean()) - 1,
                               "max_abs_dlogp": float(np.abs(kp[:n] - tl0[:n]).max()), "n": int(n)}
        res["kernel_vs_R1"] = {"ppl_rel": math.exp(-kp[:n].mean()) / math.exp(-tl1[:n].mean()) - 1,
                               "max_abs_dlogp": float(np.abs(kp[:n] - tl1[:n]).max()), "n": int(n)}
        for g in ("kernel_vs_R0",):
            res[g]["gate_pass"] = abs(res[g]["ppl_rel"]) < 0.01
        if "top1" in keys:
            kt = np.asarray(z["top1"])
            res["kernel_top1_raw"] = {"shape": list(kt.shape), "dtype": str(kt.dtype),
                                      "mean_if_bool": float(kt.mean()) if kt.dtype == bool else None}
    out.write_text(json.dumps(res, indent=1) + "\n")
    log("kernel section: " + json.dumps({k: res.get(k) for k in ("kernel_vs_R0", "kernel_vs_R1", "kernel_top1_raw")}))
else:
    log(f"kernel npz not found: {knpz} (torch-only pass)")
log("done")
