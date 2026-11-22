"""Mid-8L megakernel validation target (rec-pp2).

Correctness reference for a Flint mid-8L decode megakernel (Gemini Phase B,
rec/gemini-kernels). The kernel is judged against the torch model layer
(rec_mlx.torch_rocm, the REC_BACKEND=rocm path; it is pinned to the MLX
production model by engine/tests/test_torch_rocm.py) run in fp32 and
teacher-forced along the kernel's own trajectory. See README.md.

Subcommands
  cases    print the fixed case matrix (JSON) for kernel drivers
  produce  stand-in "kernel": torch layer decode -> kernel-output npz
           (--impl torch-fp32|torch-fp32-bf16kv|torch-bf16, --fault for negative controls)
  check    validate one kernel-output npz           (exit 0 = PASS)
  matrix   validate every case in a directory        (exit 0 = all PASS)

Kernel-output npz schema "rec-mk-out/1":
  schema       str   "rec-mk-out/1"
  case         str   case name from `cases`
  ckpt_sha256  str   sha256 of the checkpoint's model-00001.safetensors
  impl         str   free text (kernel name / build)
  prompt       int32 [P]       must equal the case prompt
  tokens       int32 [n_out]   greedy tokens, tokens[i] chosen from step_logits[i]
  last_logits  f32   [V]       logits after consuming tokens[-1] (picks token n_out+1)
  step_logits  f32   [n_out,V] logits that picked tokens[i] (required)
  step_experts int32 [P+n_out, layers, k] optional: the kernel's chosen experts at every
                     position (dense layers -1). When present, the router near-tie rule
                     (README 4a) uses it instead of searching.
  kv_dtype     str   optional "fp32" (default) | "bf16": KV-cache storage precision.
                     Selects the reference: fp32 compute with K/V rounded to this
                     dtype at the storage points. Compute must be fp32 either way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCHEMA = "rec-mk-out/1"
# Pass criteria (README.md "Pass criteria").
# max|kernel - ref| / max|ref| over last_logits and every step_logits row, by the
# declared KV storage dtype. Measured on mid-8L (dusk 890M): correct fp32 floor
# 3.4e-6, correct fp32-compute/bf16-KV floor 5.0e-4. The softmax-mix MoE fault
# keeps tokens exact and lands at 4.2e-4..9.0e-4 vs the bf16-KV reference
# (invisible) but 2.5e-4..1.05e-3 vs fp32 (caught), so sign-off needs both
# builds (README "Pass criteria").
LOGIT_RTOL = {"fp32": 1e-4, "bf16": 1e-3}
TIE_TOL = 5e-3      # a token may differ from the fp32 argmax only if the fp32 margin <= this
# MoE router near-tie substitution (README 4a, adopted 2026-11-19), bf16-KV tier
# only: one k-th/(k+1)-th expert swap at a selection-score gap <= this, at or
# before the first failing step, then the full unchanged criteria. PROVISIONAL:
# rests on one observed flip (gap 4.1e-5); recalibration is bead rec-tan.3.
ROUTER_TIE_TOL = 2.5e-4
ROUTER_TIE_TIERS = ("bf16",)
MAX_ROUTER_SWAPS = 1
MAX_ROUTER_CANDIDATES = 8  # bound on replays tried (each is a single swap)
# step_experts path policy. "all-declared" = ADOPTED (rec-tan.4, 2026-11-19; README 4c):
# every kernel-declared deviation must be a k/(k+1) exchange at gap <= ROUTER_TIE_TOL,
# then the full criteria against the reference forced to the kernel's routing; in the
# fp32-KV tier any declared deviation is a FAIL. "single" = superseded rule (one
# deviation at or before the first failing row), kept selectable for comparison.
# Outputs without step_experts use the single-swap search (unchanged).
ROUTER_EXPERTS_POLICY = "all-declared"
N_OUT = 24


def case_matrix(vocab: int) -> dict[str, dict]:
    """Fixed cases. Token ids >= 6 are plain 'tokN' (rec_mlx.synthetic tokenizer:
    0 pad, 1 bos, 2 eos, 3 system, 4 user, 5 assistant)."""
    def tok(n: int) -> int:  # plain token id valid for any vocab
        return 6 + n % (vocab - 6)

    rng = np.random.default_rng(0)
    wrap = [1] + [int(t) for t in rng.integers(6, vocab, size=511)]
    return {
        # short prompt, decode only inside the first window
        "short4": {"prompt": [1, tok(37), tok(1000), tok(52000)], "n_out": N_OUT},
        # chat-template-shaped ids: bos, <|user|>, text, <|assistant|>
        "chat": {"prompt": [1, 4, tok(777), tok(8888), 5], "n_out": N_OUT},
        # 512-token prompt + 24 decode -> positions 0..535 > 513: exercises the
        # sliding-window ring wrap (slot = pos % 513) and long full-attention KV
        "wrap512": {"prompt": wrap, "n_out": N_OUT},
    }


def shard_sha256(ckpt: Path) -> str:
    h = hashlib.sha256()
    with open(ckpt / "model-00001.safetensors", "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def vocab_of(ckpt: Path) -> int:
    return json.loads((ckpt / "config.json").read_text())["vocab_size"]


# ---------------------------------------------------------------- reference
class Reference:
    """fp32-compute torch layer, teacher-forced, K/V stored at `kv_dtype` (exact
    rounding at the contract's storage points). Loaded once; each run can record
    the MoE top-k boundary gap at every (layer, position) and force one swap."""

    def __init__(self, ckpt: Path, device: str, kv_dtype: str = "fp32"):
        import torch

        from rec_mlx.torch_rocm import load_model

        # REC_REF_MMAP=1: map the shards instead of reading them (full Flint on a 46 GB host, rec-6ek)
        self.model = load_model(ckpt, device=device, dtype=torch.float32,
                                mmap=os.environ.get("REC_REF_MMAP") == "1")
        self.model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[kv_dtype]
        self.k = self.model.cfg.topk
        self.scores: dict[int, np.ndarray] = {}  # last run: layer -> selection scores [L, E]

    def run(self, prompt: list[int], tokens: list[int], swaps: list[dict] | None = None,
            force: np.ndarray | None = None):
        """Row i (i < n) = logits that should pick tokens[i]; row n = after tokens[-1].
        Returns (logits [n+1, V], gaps). gaps: one dict per (layer, position) with the
        selection-score gap between rank k and rank k+1 and those two experts.
        swaps = [{"layer", "pos", "drop", "add"}]: the reference takes `add` instead of
        `drop` at that (layer, absolute position). The gate uses at most one; more
        are accepted here so tests can build multi-flip negative controls.
        force = int [P+n, layers, k]: use exactly these experts at every MoE
        (layer, position) (kernel-emitted `step_experts`); gaps still describe the
        reference's own ranking, so each boundary is compared on a trajectory
        whose upstream routing equals the kernel's."""
        import types

        import torch
        import torch.nn.functional as F

        model, K, gaps, swaps = self.model, self.k, [], swaps or []
        self.scores = {}
        cfg = model.cfg
        self.used = np.full((len(prompt) + len(tokens), cfg.layers, K), -1, np.int32)  # experts actually used
        real_topk = torch.topk

        def moe(m, x, i):
            pre = f"model.layers.{i}.mlp"
            sel = torch.sigmoid(F.linear(x.float(), m.t[f"{pre}.gate.weight"].float())) \
                + m.t[f"{pre}.gate.expert_bias"].float().view(1, 1, -1)  # same as _moe
            self.scores[i] = sel[0].cpu().numpy()
            top = real_topk(sel[0], K + 1, dim=-1)
            for p in range(sel.shape[1]):
                gaps.append({"layer": i, "pos": p, "gap": float(top.values[p, K - 1] - top.values[p, K]),
                             "kth": int(top.indices[p, K - 1]), "next": int(top.indices[p, K]),
                             "top": sorted(int(e) for e in top.indices[p, :K])})
            mine = [sw for sw in swaps if sw["layer"] == i]

            def topk(t, k, dim=-1, **kw):
                r = real_topk(t, k, dim=dim, **kw)
                if k != K or t.dim() != 3:
                    return r
                idx = r.indices.clone()
                if force is not None:
                    idx[0] = torch.as_tensor(force[: idx.shape[1], i, :], device=idx.device, dtype=idx.dtype)
                for sw in mine:
                    if sw["pos"] >= idx.shape[1]:  # position not in this sequence (yet)
                        continue
                    row = idx[0, sw["pos"]]
                    if sw["drop"] not in row.tolist() or sw["add"] in row.tolist():
                        raise ValueError(f"swap {sw} does not apply to {row.tolist()}")
                    row[row == sw["drop"]] = sw["add"]
                self.used[: idx.shape[1], i] = idx[0].cpu().numpy()
                return torch.return_types.topk((torch.gather(t, dim, idx), idx))

            torch.topk = topk
            try:
                return type(m)._moe(m, x, i)
            finally:
                torch.topk = real_topk

        model._moe = types.MethodType(moe, model)
        try:
            with torch.inference_mode():
                ids = torch.tensor([prompt + tokens], device=model.device)
                out = model.forward(ids)[0, len(prompt) - 1:]
        finally:
            del model._moe
        return out.float().cpu().numpy(), gaps


def reference_logits(ckpt: Path, prompt: list[int], tokens: list[int], device: str,
                     kv_dtype: str = "fp32"):
    """Teacher-forced reference logits (see Reference.run)."""
    return Reference(ckpt, device, kv_dtype).run(prompt, tokens)[0]


def _evaluate(d, ref: np.ndarray, kv_dtype: str, vocab: int) -> dict:
    """The unchanged section-4 criteria for one kernel output against one reference."""
    tokens = d["tokens"].tolist()
    rtol = LOGIT_RTOL[kv_dtype]
    problems, flips, bad = [], [], []
    for i, tok in enumerate(tokens):
        best = int(ref[i].argmax())
        if best != tok:
            margin = float(ref[i, best] - ref[i, tok])
            (flips if margin <= TIE_TOL else bad).append({"step": i, "ref": best, "kernel": tok,
                                                          "margin": margin})
    kl = d["last_logits"].astype(np.float64)
    last_rel = float(np.abs(kl - ref[-1]).max() / max(np.abs(ref[-1]).max(), 1e-6))
    step_rel, rows, structural = None, None, False
    sl = d["step_logits"].astype(np.float64)
    if sl.shape != (len(tokens), vocab):
        problems.append(f"step_logits shape {sl.shape} != ({len(tokens)}, {vocab})")
        structural = True
    else:
        rows = np.abs(sl - ref[:-1]).max(-1) / np.maximum(np.abs(ref[:-1]).max(-1), 1e-6)
        step_rel = float(rows.max())
        if not np.isfinite(sl).all():
            problems.append("non-finite step_logits")
            structural = True
    if not np.isfinite(kl).all():
        problems.append("non-finite last_logits")
        structural = True
    if bad:
        problems.append(f"{len(bad)} token(s) differ from the fp32 reference beyond the near-tie margin")
    if last_rel > rtol:
        problems.append(f"last_logits rel err {last_rel:.3e} > {rtol} ({kv_dtype}-KV tier)")
    if step_rel is not None and step_rel > rtol:
        problems.append(f"step_logits max rel err {step_rel:.3e} > {rtol} ({kv_dtype}-KV tier)")
    # first failing row: a bad token or a row over the tier; row n = last_logits
    fails = [b["step"] for b in bad] + ([int(np.argmax(rows > rtol))] if rows is not None and
                                         (rows > rtol).any() else [])
    first_fail = min(fails) if fails else (len(tokens) if last_rel > rtol else None)
    return {"problems": problems, "flips": flips, "bad": bad, "last_rel": last_rel,
            "step_rel": step_rel, "first_fail": first_fail, "structural": structural}


def step_experts_of(d, reference: Reference, total: int) -> tuple[np.ndarray | None, list[str]]:
    """Optional rec-mk-out/1 key step_experts int32 [P+n_out, layers, k]: the kernel's
    chosen experts at every position (dense layers -1). Returns (array | None, problems)."""
    if "step_experts" not in d.files:
        return None, []
    se = np.asarray(d["step_experts"])
    cfg = reference.model.cfg
    want = (total, cfg.layers, cfg.topk)
    if se.shape != want:
        return None, [f"step_experts shape {se.shape} != {want}"]
    moe = [i for i in range(cfg.layers) if i not in cfg.dense_layers]
    probs = []
    if (se[:, [i for i in range(cfg.layers) if i in cfg.dense_layers]] != -1).any():
        probs.append("step_experts: dense layers must be -1")
    m = se[:, moe]
    if (m < 0).any() or (m >= cfg.num_experts).any():
        probs.append("step_experts: expert id outside [0, experts)")
    elif any(len(set(r)) != cfg.topk for r in m.reshape(-1, cfg.topk).tolist()):
        probs.append("step_experts: repeated expert within a top-k row")
    return (None if probs else se), probs


def routing_deviations(reference: Reference, prompt, tokens, step_experts) -> tuple[list[dict], np.ndarray]:
    """Clean per-boundary comparison: run the reference with the kernel's routing
    forced everywhere, and list each (layer, pos) where the kernel's expert set
    differs from the reference's own top-k there. Upstream routing is identical by
    construction, so no deviation is a downstream consequence of another."""
    ref_f, gaps = reference.run(prompt, tokens, force=step_experts)
    dev = []
    for g in gaps:
        ks = sorted(int(e) for e in step_experts[g["pos"], g["layer"]])
        if ks != g["top"]:
            exch = sorted(set(g["top"]) - {g["kth"]} | {g["next"]}) == ks
            dev.append({**{k: g[k] for k in ("layer", "pos", "gap", "kth", "next")},
                        "kernel": ks, "ref": g["top"], "k_k1_exchange": exch})
    return dev, ref_f


def router_substitution_from_experts(reference: Reference, d, prompt, tokens, kv_dtype, vocab, ev,
                                     step_experts):
    """README 4a with kernel-emitted routing (preferred path): the kernel names the
    swap, so nothing is searched. Same limits as the search path: bf16-KV tier, the
    only routing deviation at or before the first failing row, a k-th/(k+1)-th
    exchange at gap <= ROUTER_TIE_TOL; then the full criteria against the reference
    with that single swap. Returns (router_record | None, ref, ev, deviations, note)."""
    dev, ref_f = routing_deviations(reference, prompt, tokens, step_experts)
    p0 = len(prompt) - 1
    if ROUTER_EXPERTS_POLICY == "all-declared":
        return _all_declared(d, kv_dtype, vocab, ev, dev, ref_f, p0)
    if not ev["problems"] or ev["structural"] or kv_dtype not in ROUTER_TIE_TIERS:
        return None, None, None, dev, None
    early = [x for x in dev if x["pos"] <= p0 + ev["first_fail"]]
    if not early:
        return None, None, None, dev, "no kernel routing deviation at or before the first failing row"
    if len(early) > MAX_ROUTER_SWAPS:
        return None, None, None, dev, f"{len(early)} routing deviations at or before the first failing row"
    x = early[0]
    if not x["k_k1_exchange"]:
        return None, None, None, dev, f"routing deviation L{x['layer']}@{x['pos']} is not a k/(k+1) exchange"
    if x["gap"] > ROUTER_TIE_TOL:
        return None, None, None, dev, (f"routing deviation L{x['layer']}@{x['pos']} at gap {x['gap']:.2e} "
                                       f"> ROUTER_TIE_TOL {ROUTER_TIE_TOL}")
    swap = {"layer": x["layer"], "pos": x["pos"], "drop": x["kth"], "add": x["next"]}
    ref_s, _ = reference.run(prompt, tokens, [swap])
    ev_s = _evaluate(d, ref_s, kv_dtype, vocab)
    if ev_s["problems"]:
        return None, None, None, dev, "substituted reference still fails: " + "; ".join(ev_s["problems"])
    rec = {"kind": "router", "source": "step_experts", **swap, "step": x["pos"] - p0, "gap": x["gap"]}
    return rec, ref_s, ev_s, dev, None


def _all_declared(d, kv_dtype, vocab, ev, dev, ref_f, p0):
    """Adopted policy (README 4c, rec-tan.4). Returns like router_substitution_from_experts; a
    returned ev with problems signals FAIL even if the stock evaluation passed."""
    if ev["structural"]:
        return None, None, None, dev, None
    if dev and kv_dtype not in ROUTER_TIE_TIERS:
        fail = dict(ev, problems=ev["problems"] + [f"{len(dev)} routing deviation(s) in the {kv_dtype}-KV tier"])
        return None, None, fail, dev, "the fp32-KV tier allows no routing deviation"
    if not ev["problems"] and not dev:
        return None, None, None, dev, None
    bad = [x for x in dev if not x["k_k1_exchange"] or x["gap"] > ROUTER_TIE_TOL]
    if bad:
        x = bad[0]
        note = (f"routing deviation L{x['layer']}@{x['pos']} gap {x['gap']:.2e} "
                f"({'k/(k+1)' if x['k_k1_exchange'] else 'not a k/(k+1) exchange'}) exceeds the rule")
        return None, None, dict(ev, problems=ev["problems"] or [note]), dev, note
    ev_f = _evaluate(d, ref_f, kv_dtype, vocab)
    if ev_f["problems"]:
        return None, None, ev_f, dev, "reference forced to the kernel's routing still fails"
    rec = {"kind": "router", "source": "step_experts", "policy": "all-declared", "n": len(dev),
           "deviations": [{k: x[k] for k in ("layer", "pos", "gap", "kth", "next")} for x in dev]}
    if dev:
        rec.update({k: dev[0][k] for k in ("layer", "pos", "gap")})
        rec.update(drop=dev[0]["kth"], add=dev[0]["next"], step=dev[0]["pos"] - p0)
    return (rec if dev else None), ref_f, ev_f, dev, None


def router_substitution(reference: Reference, d, prompt, tokens, kv_dtype, vocab, ev, gaps):
    """README 4a (adopted): if the stock evaluation fails the bf16-KV tier, try single
    k-th/(k+1)-th swaps at reference gaps <= ROUTER_TIE_TOL, at or before the first
    failing row, smallest gap first; accept the first whose substituted reference meets
    the full unchanged criteria. Returns (router_record | None, n_tried, ref, ev)."""
    if not ev["problems"] or ev["structural"] or kv_dtype not in ROUTER_TIE_TIERS:
        return None, 0, None, None
    p0 = len(prompt) - 1
    last_pos = p0 + ev["first_fail"]
    cands = sorted((g for g in gaps if g["gap"] <= ROUTER_TIE_TOL and g["pos"] <= last_pos),
                   key=lambda g: g["gap"])[:MAX_ROUTER_CANDIDATES]
    for n, g in enumerate(cands, 1):
        swap = {"layer": g["layer"], "pos": g["pos"], "drop": g["kth"], "add": g["next"]}
        ref_s, _ = reference.run(prompt, tokens, [swap])  # MAX_ROUTER_SWAPS == 1
        ev_s = _evaluate(d, ref_s, kv_dtype, vocab)
        if not ev_s["problems"]:
            return {"kind": "router", **swap, "step": g["pos"] - p0, "gap": g["gap"]}, n, ref_s, ev_s
    return None, len(cands), None, None


def check(ckpt: Path, path: Path, device: str, sha: str | None = None, verbose: bool = True,
          ref_kv: str = "auto", spot: bool = False) -> dict:
    """spot=True (rec-6ek, in-window full-Flint spot): accept a strict PREFIX of the case's n_out tokens (the
    window emits a short run to bound launch time); every other check and the tier are unchanged."""
    d = np.load(path, allow_pickle=False)
    problems: list[str] = []
    for key in ("schema", "case", "ckpt_sha256", "prompt", "tokens", "last_logits", "step_logits"):
        if key not in d.files:
            problems.append(f"missing key {key}")
    if problems:
        return _report(path, {"pass": False, "problems": problems}, verbose)
    vocab = vocab_of(ckpt)
    cases = case_matrix(vocab)
    case = str(d["case"])
    if str(d["schema"]) != SCHEMA:
        problems.append(f"schema {d['schema']!s} != {SCHEMA}")
    if case not in cases:
        problems.append(f"unknown case {case}")
    sha = sha or shard_sha256(ckpt)
    if str(d["ckpt_sha256"]) != sha:
        problems.append("ckpt_sha256 does not match the checkpoint (wrong weights?)")
    prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
    if case in cases:
        if prompt != cases[case]["prompt"]:
            problems.append("prompt differs from the case prompt")
        n_want = cases[case]["n_out"]
        if (len(tokens) != n_want) if not spot else not (0 < len(tokens) <= n_want):
            problems.append(f"{len(tokens)} tokens, case wants {n_want}" + (" (spot: 1..n_out)" if spot else ""))
    if d["last_logits"].shape != (vocab,):
        problems.append(f"last_logits shape {d['last_logits'].shape} != ({vocab},)")
    if any(not 0 <= t < vocab for t in tokens):
        problems.append("token id outside the vocab")
    if problems:
        return _report(path, {"case": case, "pass": False, "problems": problems}, verbose)

    kv_dtype = ref_kv if ref_kv != "auto" else (str(d["kv_dtype"]) if "kv_dtype" in d.files else "fp32")
    if kv_dtype not in ("fp32", "bf16"):
        return _report(path, {"case": case, "pass": False, "problems": [f"kv_dtype {kv_dtype!r}"]}, verbose)
    reference = Reference(ckpt, device, kv_dtype)
    result = check_arrays(reference, d, prompt, tokens, kv_dtype, vocab, case)
    if spot:
        result["spot_prefix"] = f"{len(tokens)}/{cases[case]['n_out']}"
    return _report(path, result, verbose)


def check_arrays(reference: Reference, d, prompt: list[int], tokens: list[int], kv_dtype: str, vocab: int,
                 case: str) -> dict:
    """The spec (a) verdict for one kernel output against an already-loaded reference (body of `check`,
    shared with the rec-1f4 quality gate so one Reference serves many items)."""
    ref, gaps = reference.run(prompt, tokens)
    ev = _evaluate(d, ref, kv_dtype, vocab)
    stock_problems = list(ev["problems"])
    se, se_problems = step_experts_of(d, reference, len(prompt) + len(tokens))
    if se_problems:
        return {"case": case, "pass": False, "problems": se_problems}
    deviations, note, tried = None, None, 0
    if se is not None:  # preferred: the kernel names its routing; no search
        router, ref_s, ev_s, deviations, note = router_substitution_from_experts(
            reference, d, prompt, tokens, kv_dtype, vocab, ev, se)
    else:
        router, tried, ref_s, ev_s = router_substitution(reference, d, prompt, tokens, kv_dtype, vocab, ev, gaps)
    if router:
        ref, ev = ref_s, ev_s
    elif ev_s is not None and ev_s["problems"]:  # all-declared policy: a failing verdict
        ev = ev_s
    flips, bad, problems = ev["flips"], ev["bad"], ev["problems"]
    last_rel, step_rel = ev["last_rel"], ev["step_rel"]
    ref_tokens = [int(r.argmax()) for r in ref[:-1]]
    scale = float(np.abs(ref).max())
    result = {
        "case": case, "impl": str(d["impl"]) if "impl" in d.files else "?", "ref_kv": kv_dtype,
        "logit_rtol": LOGIT_RTOL[kv_dtype],
        "pass": not problems, "problems": problems,
        "tokens_exact": not bad and not flips, "near_ties_accepted": flips + ([router] if router else []),
        "bad_tokens": bad[:5],
        "last_logits_rel": last_rel, "step_logits_max_rel": step_rel,
        "ref_logit_absmax": scale, "distinct_ref_tokens": len(set(ref_tokens)),
        "router_near_tie": router, "router_candidates_tried": tried,
        "min_router_gap": min((g["gap"] for g in gaps), default=None),
    }
    result["router_path"] = "step_experts" if se is not None else "search"
    if deviations is not None:
        result["routing_deviations"] = deviations[:10]
        result["n_routing_deviations"] = len(deviations)
    if note and problems:
        result["router_note"] = note
    if router:
        result["stock_problems"] = stock_problems
    if len(set(ref_tokens)) <= 2:
        result["warning"] = "reference trajectory nearly constant: degenerate case"
    result["_ref"] = ref  # final reference rows used for the verdict (popped by callers that serialise)
    return result


def _report(path, result, verbose):
    result.pop("_ref", None)
    if verbose:
        print(json.dumps({"file": str(path), **result}, indent=1))
    return result


# ---------------------------------------------------------------- stand-in producer
FAULTS = ("none", "no-silu", "softmax-mix")


def _apply_fault(model, fault: str):
    """Negative controls: bug classes seen in this port (rec-qib / retired tiny reference).

    no-silu      gate activation dropped (dense, routed and shared experts)
    softmax-mix  MoE mixing weights = softmax(scores[top]) instead of the
                 renormalised sigmoid scores
    (expert_bias-inside-sigmoid is not offered: production-format checkpoints
    have no expert_bias, so that bug cannot show on this target.)
    """
    import types

    import torch
    import torch.nn.functional as F

    from rec_mlx import torch_rocm

    if fault == "no-silu":
        shim = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
        shim.silu = lambda x: x

        def forward(self, *args, **kwargs):  # scoped to this model's forward calls
            torch_rocm.F = shim
            try:
                return type(self).forward(self, *args, **kwargs)
            finally:
                torch_rocm.F = F
        model.forward = types.MethodType(forward, model)
    elif fault == "softmax-mix":
        real_gather = torch.gather

        def moe(self, x, i):
            def gather(src, dim, idx):  # only the router's mixing-weight gather is altered
                return torch.softmax(real_gather(src, dim, idx), dim=-1)
            torch.gather = gather
            try:
                return type(self)._moe(self, x, i)
            finally:
                torch.gather = real_gather
        model._moe = types.MethodType(moe, model)
    return model


def produce(ckpt: Path, case: str, out: Path, impl: str, device: str, fault: str = "none") -> Path:
    import torch

    from rec_mlx.torch_rocm import load_model

    dtype = {"torch-fp32": torch.float32, "torch-fp32-bf16kv": torch.float32,
             "torch-bf16": torch.bfloat16}[impl]
    spec = case_matrix(vocab_of(ckpt))[case]
    model = _apply_fault(load_model(ckpt, device=device, dtype=dtype), fault)
    kv_dtype = "bf16" if impl in ("torch-fp32-bf16kv", "torch-bf16") else "fp32"
    if impl == "torch-fp32-bf16kv":
        model.kv_store_dtype = torch.bfloat16  # fp32 compute, bf16 KV storage
    prompt = spec["prompt"]
    with torch.inference_mode():  # kernel-shaped flow: prefill, then one token per step
        kv = model.new_cache()
        logits = model.forward(torch.tensor([prompt], device=model.device), kv)[0, -1]
        tokens, steps = [], []
        for _ in range(spec["n_out"]):
            steps.append(logits.float().cpu().numpy())
            tokens.append(int(logits.argmax()))
            logits = model.forward(torch.tensor([[tokens[-1]]], device=model.device), kv)[0, -1]
    np.savez(out, schema=np.array(SCHEMA), case=np.array(case), ckpt_sha256=np.array(shard_sha256(ckpt)),
             impl=np.array(f"{impl}@{device}" + (f" fault={fault}" if fault != "none" else "")),
             kv_dtype=np.array(kv_dtype),
             prompt=np.array(prompt, np.int32), tokens=np.array(tokens, np.int32),
             last_logits=logits.float().cpu().numpy(), step_logits=np.stack(steps))
    return out


# ---------------------------------------------------------------- CLI
# rec-s3m (adopted 2026-11-20): the sign-off reference is torch fp32 on the CPU -- the same
# reference rec-ft9 adopted for the engine e2e. The bf16-KV tier is reference-device sensitive
# (albion R9700 torch 2.10/rocm7 reference: short4 1.056e-3 FAIL vs 9.24e-4 PASS on CPU for the
# same kernel outputs), so only CPU-reference matrices count as sign-off; any other device is a
# diagnostic run (exit 2 when it would otherwise pass).
SIGNOFF_REF_DEVICE = "cpu"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--ckpt", type=Path, required=True)
    p = sub.add_parser("produce")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--impl", choices=("torch-fp32", "torch-fp32-bf16kv", "torch-bf16"), default="torch-bf16")
    p.add_argument("--fault", choices=FAULTS, default="none")
    p.add_argument("--device", default=None)
    k = sub.add_parser("check")
    k.add_argument("--ckpt", type=Path, required=True)
    k.add_argument("kernel_out", type=Path)
    k.add_argument("--ref-device", default=SIGNOFF_REF_DEVICE,
                   help="reference device; sign-off requires %(default)s (rec-s3m)")
    k.add_argument("--ref-kv", choices=("auto", "fp32", "bf16"), default="auto")
    m = sub.add_parser("matrix")
    m.add_argument("--ckpt", type=Path, required=True)
    m.add_argument("kernel_dir", type=Path)
    m.add_argument("--ref-device", default=SIGNOFF_REF_DEVICE,
                   help="reference device; sign-off requires %(default)s (rec-s3m)")
    m.add_argument("--ref-kv", choices=("auto", "fp32", "bf16"), default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        print(json.dumps({"schema": SCHEMA, "ckpt_sha256": shard_sha256(a.ckpt),
                          "cases": case_matrix(vocab_of(a.ckpt))}))
        return 0
    if a.cmd == "produce":
        print(produce(a.ckpt, a.case, a.out, a.impl, a.device or _default_device(), a.fault))
        return 0
    device = a.ref_device
    signoff = device == SIGNOFF_REF_DEVICE
    if a.cmd == "check":
        ok = check(a.ckpt, a.kernel_out, device, ref_kv=a.ref_kv)["pass"]
        print(f"ref=torch-fp32@{device} sign-off={'yes' if signoff else 'NO (diagnostic, rec-s3m)'}")
        return (0 if signoff else 2) if ok else 1
    sha = shard_sha256(a.ckpt)
    files = sorted(a.kernel_dir.glob("*.npz"))
    results = [check(a.ckpt, f, device, sha, verbose=False, ref_kv=a.ref_kv) for f in files]
    for f, r in zip(files, results):
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {f.name:34} case={r.get('case', '?'):8} "
              f"ref=torch-fp32@{device} ref_kv={r.get('ref_kv', '?'):4} "
              f"tokens_exact={r.get('tokens_exact')} ties={len(r.get('near_ties_accepted', []))} "
              f"last_rel={r.get('last_logits_rel', float('nan')):.2e} "
              f"step_rel={r.get('step_logits_max_rel') or float('nan'):.2e}  {'; '.join(r['problems'])}"
              + (f"router-tie L{rt['layer']}@pos{rt['pos']} {rt['drop']}->{rt['add']} gap {rt['gap']:.2e}"
                 if (rt := r.get("router_near_tie")) else ""))
    # sign-off coverage (adopted policy): every case must pass in BOTH KV tiers
    covered = {(r.get("case"), r.get("ref_kv")) for r in results if r["pass"]}
    missing = {(c, t) for c in case_matrix(vocab_of(a.ckpt)) for t in LOGIT_RTOL} - covered
    print(f"{sum(r['pass'] for r in results)}/{len(results)} PASS; (case, kv tier) without a passing run: "
          f"{sorted(missing) or 'none'}; reference torch-fp32@{device}; sign-off run: "
          f"{'yes' if signoff else f'NO (diagnostic; sign-off reference is {SIGNOFF_REF_DEVICE}, rec-s3m)'}")
    ok = bool(results) and all(r["pass"] for r in results) and not missing
    return (0 if signoff else 2) if ok else 1


if __name__ == "__main__":
    sys.exit(main())
