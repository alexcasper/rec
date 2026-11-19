"""Conformance for rec_mlx.torch_rocm (rec-9na). Run on a torch device (ROCm/CUDA/CPU).

  python engine/tests/torch_rocm_tiny_check.py --ckpt <ckpt_dir> [--device cuda]
                                               [--fp32-npz <tiny_flint_fp32.npz>]

Checks, each printed as PASS/FAIL with the number:
  1. dequant parity: our INT4/INT8 unpack == mx.dequantize output in the fp32 npz
  2. KV-cache consistency: prefill+incremental decode logits == full recompute
  3. device vs CPU: fp32-on-device drift, then bf16 teacher-forced near-tie check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec_mlx.torch_rocm import load_model  # noqa: E402


def check_dequant(model, npz_path: Path) -> bool:
    ref = np.load(npz_path)
    worst = 0.0
    n = 0
    for key in ref.files:
        base = key.replace("__", ".")  # export stores dequantized tensors under the base name
        if base in model.q:
            ours = model.q[base].dense(torch.float32).cpu().numpy()
        elif base == "model.embed_tokens":
            ours = model.embed.dense(torch.float32).cpu().numpy()
        else:
            continue
        theirs = ref[key]
        if ours.shape != theirs.shape:
            # fp32 npz stores switch_mlp as [E, out, K] already; ours too. Mismatch is a bug.
            print(f"  shape mismatch {base}: ours {ours.shape} vs ref {theirs.shape}")
            return False
        worst = max(worst, float(np.abs(ours - theirs).max()))
        n += 1
    ok = n > 0 and worst < 1e-5
    print(f"1 dequant parity: {n} tensors, max|diff|={worst:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


@torch.no_grad()
def check_kv(model, device) -> bool:
    torch.manual_seed(0)
    ids = torch.randint(0, model.cfg.vocab, (1, 24), device=device)
    full = model.forward(ids)
    kv = model.new_cache()
    pre = model.forward(ids[:, :16], kv)
    steps = [pre[:, -1:]]
    for t in range(16, 24):
        steps.append(model.forward(ids[:, t:t + 1], kv))
    inc = torch.cat([pre] + steps[1:], dim=1)
    d = float((inc - full).abs().max())
    amatch = bool((inc.argmax(-1) == full.argmax(-1)).all())
    ok = d < 1e-3 and amatch
    print(f"2 kv-cache consistency (bf/fp32 as configured): max|diff|={d:.3e} argmax-all-match={amatch}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


@torch.no_grad()
def greedy(model, ids, n_new: int, device) -> tuple[list[int], torch.Tensor]:
    kv = model.new_cache()
    logits = model.forward(ids, kv)
    first = logits[0, -1]
    toks = []
    tok = int(first.argmax())
    toks.append(tok)
    for _ in range(n_new - 1):
        logits = model.forward(torch.tensor([[tok]], device=device), kv)
        tok = int(logits[0, -1].argmax())
        toks.append(tok)
    return toks, first


# bf16 activations legitimately flip argmax near-ties (measured on dusk 890M:
# bf16 logit rms err 2.3e-2 vs fp32, identical on CPU and GPU); a kernel bug
# shows up as a flip with a large fp32 margin or as device-fp32 drift.
BF16_TIE_TOL = 5e-2


@torch.no_grad()
def check_device_vs_cpu(ckpt: Path, device: str, n_new: int) -> bool:
    """3a: device fp32 == cpu fp32 (isolates device bugs from dtype effects).
    3b: device bf16 teacher-forced along the cpu-fp32 greedy trajectory; every
    disagreement must be a near-tie (fp32 margin <= BF16_TIE_TOL)."""
    ref = load_model(ckpt, device="cpu", dtype=torch.float32)
    prompt = torch.arange(1, 33).unsqueeze(0)
    toks, _ = greedy(ref, prompt, n_new, "cpu")
    full = torch.cat([prompt, torch.tensor([toks])], 1)
    truth = ref.forward(full)[0]
    d32 = load_model(ckpt, device=device, dtype=torch.float32).forward(full.to(device))[0].cpu()
    drift = float((d32 - truth).abs().max())
    scale = max(1.0, float(truth.abs().max()))
    ok32 = drift < 1e-4 * scale
    print(f"3a device(fp32) vs cpu(fp32) {full.shape[1]} pos: max|diff|={drift:.3e} -> {'PASS' if ok32 else 'FAIL'}")
    d16 = load_model(ckpt, device=device, dtype=torch.bfloat16).forward(full.to(device))[0].cpu()
    base = prompt.shape[1] - 1
    flips, bad = 0, []
    for i, tok in enumerate(toks):
        got = int(d16[base + i].argmax())
        if got != tok:
            margin = float(truth[base + i, tok] - truth[base + i, got])
            flips += 1
            if margin > BF16_TIE_TOL:
                bad.append((i, tok, got, margin))
    rms = float(((d16 - truth) ** 2).mean().sqrt())
    ok16 = not bad
    print(f"3b device(bf16) teacher-forced {n_new} tok: logit rms err {rms:.3e}, argmax flips {flips} "
          f"(all near-ties <= {BF16_TIE_TOL}: {ok16}) -> {'PASS' if ok16 else 'FAIL'}")
    for i, tok, got, margin in bad:
        print(f"   step {i}: cpu {tok} dev {got} fp32 margin {margin:.3e}")
    return ok32 and ok16


def randomize_router(src: Path, dst: Path, seed: int = 0) -> Path:
    """Copy a checkpoint dir and replace the all-zero MoE router (gate.weight,
    gate.expert_bias) with seeded random values. The shipped cpu-baseline
    checkpoints have a zero router, so every expert ties at 0.5 and top-k
    selection is backend tie-breaking, not model behaviour."""
    import json as _json
    import shutil
    import struct

    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.suffix != ".safetensors":
            shutil.copy(p, dst / p.name)
    rng = np.random.default_rng(seed)
    for shard in sorted(src.glob("model-*.safetensors")):
        data = bytearray(shard.read_bytes())
        (n,) = struct.unpack("<Q", data[:8])
        header = _json.loads(bytes(data[8:8 + n]))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__" or not (name.endswith("mlp.gate.weight") or name.endswith("mlp.gate.expert_bias")):
                continue
            s, e = info["data_offsets"]
            count = (e - s) // 4
            vals = (rng.standard_normal(count) * 0.5).astype(np.float32)
            data[base + s: base + e] = vals.tobytes()
        (dst / shard.name).write_bytes(bytes(data))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-npz", type=Path, default=None)
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--randomize-router-into", type=Path, default=None,
                    help="write a copy of --ckpt with a seeded random MoE router here and test that")
    a = ap.parse_args()
    if a.randomize_router_into:
        a.ckpt = randomize_router(a.ckpt, a.randomize_router_into)
        print(f"using randomized-router checkpoint: {a.ckpt}")
    print(f"torch {torch.__version__} hip={torch.version.hip} device={a.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if a.device.startswith("cuda") else ""))
    results = []
    cpu32 = load_model(a.ckpt, device="cpu", dtype=torch.float32)
    if a.fp32_npz:
        results.append(check_dequant(cpu32, a.fp32_npz))
    results.append(check_kv(cpu32, "cpu"))
    results.append(check_device_vs_cpu(a.ckpt, a.device, a.new))
    print("OVERALL:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
