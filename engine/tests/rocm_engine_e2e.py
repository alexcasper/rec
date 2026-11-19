"""rec-ju9: Engine end-to-end on the rocm backend vs a torch fp32 reference.

    REC_TORCH_DEVICE=cuda python engine/tests/rocm_engine_e2e.py --ckpt DIR [--prompt-len 512] [--new 32]

For one checkpoint, greedy decode runs through Engine.generate_tokens on:
  cpu        REC_BACKEND=cpu (MLX-CPU, production model code): informational
  rocm-fp32  REC_BACKEND=rocm REC_TORCH_DTYPE=fp32
  rocm-bf16  REC_BACKEND=rocm (default bf16)

Reference (rec-ft9): the torch model layer in fp32 on the CPU (fp32
intermediates), teacher-forced on each engine trajectory. It is pinned to the
MLX model on F32 (synthetic-format) checkpoints by test_torch_rocm.py (measured
2.5e-6 max|diff| on syn-tiny). MLX QATModel is NOT an fp32 reference on
production-format checkpoints: with bf16 scales/biases it computes in bf16
activations (its logits come out bfloat16; up to 0.93 max|diff| vs torch fp32
on a random 201-token syn-tiny prompt, 0.081 on the rec-ft9 dusk trajectory),
so it is reported, not gated on. Gates (two-tier spirit of the rec-pp2 harness, 82dfdd4):
  (a1) rocm-fp32 tokens: a flip vs the fp32 reference is allowed only at a
       near-tie, fp32 margin <= TIE_TOL (5e-3)
  (a2) device drift: teacher-forced torch fp32 logits on the GPU vs the CPU,
       max|diff| / max|ref| <= FP32_DRIFT_RTOL (1e-4)
  (b)  rocm-bf16 tokens: every flip vs the fp32 reference must be explained by
       precision -- margin <= |bf16 - fp32| torch logit perturbation at that
       step (teacher-forced on the bf16 path), or a near-tie
Engine metrics (prefill seconds, decode tok/s) are printed as JSON. They
include the engine loop (sampling, detokenize, host logits copy), so they are
NOT the bare-model protocol of docs/benchmarks/run-definitions.md
(torch_rocm_bench.py is).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_engine(ckpt: Path, backend: str, prompt: list[int], new: int, dtype: str = "bf16") -> dict:
    from rec_mlx.engine import Engine, Sampling

    os.environ["REC_BACKEND"] = backend
    os.environ["REC_TORCH_DTYPE"] = dtype
    engine = Engine(ckpt, ckpt / "manifest.json", context_limit=len(prompt) + new + 64)
    # one short warmup request (allocators, kernels) outside the timed request
    list(engine.generate_tokens(prompt[:16], Sampling(temperature=0, max_tokens=4)))
    metrics = list(engine.generate_tokens(prompt, Sampling(temperature=0, max_tokens=new)))[-1]["metrics"]
    return {"startup": {k: engine.startup[k] for k in ("backend", "device_profile", "load_seconds")},
            "token_ids": metrics["token_ids"],
            "prefill_tok_s": round(len(prompt) / metrics["prefill_seconds"], 1),
            "decode_tok_s": round(metrics["decode_tokens_per_second"], 2),
            "decode_strategy": metrics["decode_strategy"]}


def mlx_logits(ckpt: Path, prompt: list[int], tokens: list[int]) -> np.ndarray:
    """MLX production-model logits for each generated step, teacher-forced."""
    import mlx.core as mx

    from rec_mlx.quantized import QATModel

    native = QATModel(ckpt).native
    out = native(mx.array([prompt + tokens[:-1]])).astype(mx.float32)
    return np.array(out)[0, len(prompt) - 1:]


TIE_TOL = 5e-3          # same near-tie margin as engine/validation/megakernel_mid (rec-pp2)
FP32_DRIFT_RTOL = 1e-4  # fp32-tier logit gate of the rec-pp2 harness


def torch_logits(ckpt: Path, prompt: list[int], tokens: list[int], dtype, device: str | None = None) -> np.ndarray:
    import torch

    from rec_mlx.torch_rocm import load_model
    from rec_mlx.torch_runtime import resolve_device

    model = load_model(ckpt, device=device or str(resolve_device()), dtype=dtype)
    with torch.inference_mode():
        ids = torch.tensor([prompt + tokens[:-1]], device=model.device)
        return model.forward(ids)[0, len(prompt) - 1:].float().cpu().numpy()


def flips(ref: np.ndarray, tokens: list[int]) -> list[dict]:
    out = []
    for i, tok in enumerate(tokens):
        best = int(ref[i].argmax())
        if best != tok:
            out.append({"step": i, "ref": best, "got": tok, "margin": float(ref[i, best] - ref[i, tok])})
    return out


def evaluate(ref32_on_t32: np.ndarray, gpu32_on_t32: np.ndarray, t32: list[int],
             ref32_on_t16: np.ndarray, bf16_perturbation: np.ndarray, t16: list[int]) -> dict:
    """The e2e gates (a1, a2, b) on teacher-forced logits; pure, unit-tested."""
    f32 = flips(ref32_on_t32, t32)
    f32_bad = [f for f in f32 if f["margin"] > TIE_TOL]
    drift = float(np.abs(gpu32_on_t32 - ref32_on_t32).max() / max(np.abs(ref32_on_t32).max(), 1e-6))
    f16 = flips(ref32_on_t16, t16)
    for f in f16:
        f["bf16_perturbation"] = float(bf16_perturbation[f["step"]])
        f["explained"] = f["margin"] <= max(f["bf16_perturbation"], TIE_TOL)
    return {"fp32_flips": f32, "fp32_unexplained_flips": f32_bad, "fp32_device_drift_rel": drift,
            "bf16_flips": f16,
            "pass": not f32_bad and drift <= FP32_DRIFT_RTOL and all(f["explained"] for f in f16)}


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--new", type=int, default=32)
    a = ap.parse_args()
    vocab = json.loads((a.ckpt / "config.json").read_text())["vocab_size"]
    rng = np.random.default_rng(0)
    prompt = [1] + [int(t) for t in rng.integers(3, vocab, size=a.prompt_len - 1)]

    runs = {"cpu": run_engine(a.ckpt, "cpu", prompt, a.new),
            "rocm-fp32": run_engine(a.ckpt, "rocm", prompt, a.new, "fp32"),
            "rocm-bf16": run_engine(a.ckpt, "rocm", prompt, a.new, "bf16")}
    t32, t16 = runs["rocm-fp32"]["token_ids"], runs["rocm-bf16"]["token_ids"]

    # Reference: torch fp32 on the CPU, teacher-forced on each engine trajectory (rec-ft9).
    ref32 = torch_logits(a.ckpt, prompt, t32, torch.float32, device="cpu")
    gpu32 = torch_logits(a.ckpt, prompt, t32, torch.float32)
    ref16 = torch_logits(a.ckpt, prompt, t16, torch.float32, device="cpu")
    pert = np.abs(torch_logits(a.ckpt, prompt, t16, torch.bfloat16)
                  - torch_logits(a.ckpt, prompt, t16, torch.float32)).max(-1)
    gate = evaluate(ref32, gpu32, t32, ref16, pert, t16)
    # MLX production model, informational only (bf16 scales on production-format ckpts).
    mlx32 = mlx_logits(a.ckpt, prompt, t32)
    mlx_info = {"mlx_vs_torch_fp32_max_abs": float(np.abs(mlx32 - ref32).max()),
                "mlx_flips_on_fp32_path": len(flips(mlx32, t32))}
    ok = (gate["pass"]
          and all(r["decode_strategy"] == ("torch" if k != "cpu" else "native_mlx") for k, r in runs.items())
          and all(len(r["token_ids"]) == a.new for r in runs.values()))
    common = next((i for i, (x, y) in enumerate(zip(runs["cpu"]["token_ids"], t16)) if x != y), a.new)
    print(json.dumps({
        "ckpt": str(a.ckpt), "prompt_len": len(prompt), "new": a.new,
        "torch_device": os.environ.get("REC_TORCH_DEVICE", "cuda"),
        "runs": {k: {kk: vv for kk, vv in r.items() if kk != "token_ids"} for k, r in runs.items()},
        "reference": "torch fp32 on CPU, teacher-forced (rec-ft9)",
        "tie_tol": TIE_TOL, "fp32_drift_rtol": FP32_DRIFT_RTOL,
        "fp32_identical_to_cpu_engine": t32 == runs["cpu"]["token_ids"],
        "fp32_flips": gate["fp32_flips"], "fp32_unexplained_flips": gate["fp32_unexplained_flips"],
        "fp32_device_drift_rel": gate["fp32_device_drift_rel"],
        "bf16_flips": gate["bf16_flips"], "bf16_free_running_common_prefix": common,
        "mlx_informational": mlx_info,
        "distinct_tokens_bf16": len(set(t16)),
        "bf16_perturbation_mean": float(pert.mean()), "pass": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
