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
  kv_dtype     str   optional "fp32" (default) | "bf16": KV-cache storage precision.
                     Selects the reference: fp32 compute with K/V rounded to this
                     dtype at the storage points. Compute must be fp32 either way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
def reference_logits(ckpt: Path, prompt: list[int], tokens: list[int], device: str,
                     kv_dtype: str = "fp32"):
    """fp32-compute torch-layer logits teacher-forced along `tokens`, with K/V
    stored at `kv_dtype` (exact rounding at the contract's storage points).
    Row i (i < n) = logits that should pick tokens[i]; row n = after tokens[-1]."""
    import torch

    from rec_mlx.torch_rocm import load_model

    model = load_model(ckpt, device=device, dtype=torch.float32)
    model.kv_store_dtype = {"fp32": None, "bf16": torch.bfloat16}[kv_dtype]
    with torch.inference_mode():
        ids = torch.tensor([prompt + tokens], device=model.device)
        out = model.forward(ids)[0, len(prompt) - 1:]
    return out.float().cpu().numpy()


def check(ckpt: Path, path: Path, device: str, sha: str | None = None, verbose: bool = True,
          ref_kv: str = "auto") -> dict:
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
        if len(tokens) != cases[case]["n_out"]:
            problems.append(f"{len(tokens)} tokens, case wants {cases[case]['n_out']}")
    if d["last_logits"].shape != (vocab,):
        problems.append(f"last_logits shape {d['last_logits'].shape} != ({vocab},)")
    if any(not 0 <= t < vocab for t in tokens):
        problems.append("token id outside the vocab")
    if problems:
        return _report(path, {"case": case, "pass": False, "problems": problems}, verbose)

    kv_dtype = ref_kv if ref_kv != "auto" else (str(d["kv_dtype"]) if "kv_dtype" in d.files else "fp32")
    if kv_dtype not in ("fp32", "bf16"):
        return _report(path, {"case": case, "pass": False, "problems": [f"kv_dtype {kv_dtype!r}"]}, verbose)
    ref = reference_logits(ckpt, prompt, tokens, device, kv_dtype)
    scale = float(np.abs(ref).max())
    flips, bad = [], []
    for i, tok in enumerate(tokens):
        best = int(ref[i].argmax())
        if best != tok:
            margin = float(ref[i, best] - ref[i, tok])
            (flips if margin <= TIE_TOL else bad).append({"step": i, "ref": best, "kernel": tok,
                                                          "margin": margin})
    kl = d["last_logits"].astype(np.float64)
    last_rel = float(np.abs(kl - ref[-1]).max() / max(np.abs(ref[-1]).max(), 1e-6))
    step_rel = None
    rtol = LOGIT_RTOL[kv_dtype]
    if True:  # step_logits is required (schema check above)
        sl = d["step_logits"].astype(np.float64)
        if sl.shape != (len(tokens), vocab):
            problems.append(f"step_logits shape {sl.shape} != ({len(tokens)}, {vocab})")
        else:
            rows = np.abs(sl - ref[:-1]).max(-1) / np.maximum(np.abs(ref[:-1]).max(-1), 1e-6)
            step_rel = float(rows.max())
            if not np.isfinite(sl).all():
                problems.append("non-finite step_logits")
    if not np.isfinite(kl).all():
        problems.append("non-finite last_logits")
    if bad:
        problems.append(f"{len(bad)} token(s) differ from the fp32 reference beyond the near-tie margin")
    if last_rel > rtol:
        problems.append(f"last_logits rel err {last_rel:.3e} > {rtol} ({kv_dtype}-KV tier)")
    if step_rel is not None and step_rel > rtol:
        problems.append(f"step_logits max rel err {step_rel:.3e} > {rtol} ({kv_dtype}-KV tier)")
    ref_tokens = [int(r.argmax()) for r in ref[:-1]]
    result = {
        "case": case, "impl": str(d["impl"]) if "impl" in d.files else "?", "ref_kv": kv_dtype,
        "logit_rtol": rtol,
        "pass": not problems, "problems": problems,
        "tokens_exact": not bad and not flips, "near_ties_accepted": flips, "bad_tokens": bad[:5],
        "last_logits_rel": last_rel, "step_logits_max_rel": step_rel,
        "ref_logit_absmax": scale, "distinct_ref_tokens": len(set(ref_tokens)),
    }
    if len(set(ref_tokens)) <= 2:
        result["warning"] = "reference trajectory nearly constant: degenerate case"
    return _report(path, result, verbose)


def _report(path, result, verbose):
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
    k.add_argument("--ref-device", default=None)
    k.add_argument("--ref-kv", choices=("auto", "fp32", "bf16"), default="auto")
    m = sub.add_parser("matrix")
    m.add_argument("--ckpt", type=Path, required=True)
    m.add_argument("kernel_dir", type=Path)
    m.add_argument("--ref-device", default=None)
    m.add_argument("--ref-kv", choices=("auto", "fp32", "bf16"), default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        print(json.dumps({"schema": SCHEMA, "ckpt_sha256": shard_sha256(a.ckpt),
                          "cases": case_matrix(vocab_of(a.ckpt))}))
        return 0
    if a.cmd == "produce":
        print(produce(a.ckpt, a.case, a.out, a.impl, a.device or _default_device(), a.fault))
        return 0
    device = a.ref_device or _default_device()
    if a.cmd == "check":
        return 0 if check(a.ckpt, a.kernel_out, device, ref_kv=a.ref_kv)["pass"] else 1
    sha = shard_sha256(a.ckpt)
    files = sorted(a.kernel_dir.glob("*.npz"))
    results = [check(a.ckpt, f, device, sha, verbose=False, ref_kv=a.ref_kv) for f in files]
    for f, r in zip(files, results):
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {f.name:34} case={r.get('case', '?'):8} "
              f"ref_kv={r.get('ref_kv', '?'):4} "
              f"tokens_exact={r.get('tokens_exact')} ties={len(r.get('near_ties_accepted', []))} "
              f"last_rel={r.get('last_logits_rel', float('nan')):.2e} "
              f"step_rel={r.get('step_logits_max_rel') or float('nan'):.2e}  {'; '.join(r['problems'])}")
    covered = {r.get("case") for r in results if r["pass"]}
    missing = set(case_matrix(vocab_of(a.ckpt))) - covered
    print(f"{sum(r['pass'] for r in results)}/{len(results)} PASS; cases without a passing run: "
          f"{sorted(missing) or 'none'}")
    return 0 if results and all(r["pass"] for r in results) and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
