"""Benchmark the rec-9na torch model layer per docs/benchmarks/run-definitions.md.

    python engine/tests/torch_rocm_bench.py --ckpt DIR [--device cuda] [--dtype bf16]

prefill tok/s: 512-token prefill, 3 warmup + mean of 3 timed runs.
decode tok/s : after the same prefill, 4 warmup decode steps, then 32 timed
               greedy steps; device-synchronized at both ends.
This path keeps weights INT4-packed on device and dequantizes inside every
matmul (no fused kernel yet, rec-aip), so it is NOT the dense-bf16
mid_bench_gpu.py configuration the existing reference numbers use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec_mlx.torch_rocm import load_model  # noqa: E402

PREFILL, WARM_RUNS, RUNS, DECODE_WARM, DECODE = 512, 3, 3, 4, 32


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = load_model(a.ckpt, device=a.device, dtype=dtype)
    dev = model.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, model.cfg.vocab, (1, PREFILL), generator=g).to(dev)

    times = []
    for i in range(WARM_RUNS + RUNS):
        sync(dev)
        t0 = time.perf_counter()
        logits = model.forward(ids, model.new_cache())
        sync(dev)
        if i >= WARM_RUNS:
            times.append(time.perf_counter() - t0)
    prefill = PREFILL / (sum(times) / len(times))

    kv = model.new_cache()
    tok = int(model.forward(ids, kv)[0, -1].argmax())
    for _ in range(DECODE_WARM):
        tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(DECODE):
        tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
    sync(dev)
    decode = DECODE / (time.perf_counter() - t0)

    packed = sum(q.packed.nbytes + q.scales.nbytes + q.biases.nbytes for q in model.q.values())
    packed += model.embed.packed.nbytes + model.embed.scales.nbytes + model.embed.biases.nbytes
    out = {
        "ckpt": str(a.ckpt), "layers": model.cfg.layers, "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
        "act_dtype": a.dtype, "kv_dtype": a.dtype, "weights": "int4-affine packed (int8 embed), dequant per matmul",
        "resident_weight_gib": round(packed / 2**30, 3),
        "prefill512_tok_s": round(prefill, 1), "decode32_tok_s": round(decode, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None,
        "logits_finite": bool(torch.isfinite(logits).all()),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
