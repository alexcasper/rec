"""Benchmark the rec-9na torch model layer per docs/benchmarks/run-definitions.md.

    python engine/tests/torch_rocm_bench.py --ckpt DIR [--device cuda] [--dtype bf16]

prefill tok/s: 512-token prefill, 3 warmup + mean of 3 timed runs.
decode tok/s : after the same prefill, 4 warmup decode steps, then 32 timed
               greedy steps; device-synchronized at both ends.
short-ctx    : prompt [1, 5, 6, 7] (run-definitions A3), 4 warmup decode steps,
               then 32 timed greedy steps (canonical window, mean of 3 runs).
serial-516   : 512 prefill + 4 decode steps end to end (fresh cache), 516 / mean
               wall of 3 runs; end-to-end incl. prefill, NOT 516 generated tokens.
All windows are device-synchronized (canonical protocol for non-persistent
paths; the persistent-kernel rows use the A1 differential instead).
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

    def decode_window(prompt_ids):
        kv = model.new_cache()
        tok = int(model.forward(prompt_ids, kv)[0, -1].argmax())
        for _ in range(DECODE_WARM):
            tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(DECODE):
            tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
        sync(dev)
        return time.perf_counter() - t0

    short = torch.tensor([[1, 5, 6, 7]], device=dev)
    decode_window(short)  # warmup run
    short_ctx = DECODE / (sum(decode_window(short) for _ in range(RUNS)) / RUNS)

    def serial516():
        sync(dev)
        t0 = time.perf_counter()
        kv = model.new_cache()
        tok = int(model.forward(ids, kv)[0, -1].argmax())
        for _ in range(DECODE_WARM - 1):  # 512 prefill positions + 4 decode positions
            tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
        model.forward(torch.tensor([[tok]], device=dev), kv)
        sync(dev)
        return time.perf_counter() - t0

    serial516()
    t516 = sum(serial516() for _ in range(RUNS)) / RUNS

    packed = sum(q.packed.nbytes + q.scales.nbytes + q.biases.nbytes for q in model.q.values())
    packed += model.embed.packed.nbytes + model.embed.scales.nbytes + model.embed.biases.nbytes
    out = {
        "ckpt": str(a.ckpt), "layers": model.cfg.layers, "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
        "act_dtype": a.dtype, "kv_dtype": a.dtype, "weights": "int4-affine packed (int8 embed), dequant per matmul",
        "resident_weight_gib": round(packed / 2**30, 3),
        "prefill512_tok_s": round(prefill, 1), "decode32_tok_s": round(decode, 2),
        "decode32_short_ctx_tok_s": round(short_ctx, 2), "serial_516_tok_s": round(516 / t516, 2),
        "t516_s": round(t516, 4), "protocol": "canonical sync-bounded windows",
        "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None,
        "logits_finite": bool(torch.isfinite(logits).all()),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
