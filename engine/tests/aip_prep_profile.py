"""rec-aip prep: per-op stage timing of the INT4 torch model layer on ROCm.

    python engine/tests/aip_prep_profile.py --ckpt DIR --device cuda [--tag name]

Stage timing via CUDA events around top-level model methods:
  embed        embed.rows lookup (dequant INT8 rows)
  rms_norm     all RMSNorm calls (input/post/final, q/k norms incl. in attention)
  attention    Model._attention (q/k/v/o lins + rope + softmax(sinks))
  mlp_dense    Model._mlp on dense layers (tag without 'shared_expert')
  moe          Model._moe total (router + routed experts + shared expert)
  mlp_shared   the _mlp call inside _moe (also counted inside moe — subtract)
  lm_head_dq   lm_head QLinear.dense dequant (the F.linear after it is small)
  lin_dq       ALL QLinear.dense dequant time, grouped per tensor name
  expert_dq    QLinear.expert dequant time (routed experts), per MoE layer

Views: exclusive stage shares (% of forward) + dequant sub-total per stage.
Protocol mirrors torch_rocm_bench.py (prefill 512 x3 after 3 warmups; 32
greedy decode steps after 4 warmups). Output: JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rec_mlx.torch_rocm as tr  # noqa: E402
from rec_mlx.torch_rocm import load_model  # noqa: E402

PREFILL, WARM_RUNS, RUNS, DECODE_WARM, DECODE = 512, 3, 3, 4, 32


class Timer:
    """Accumulates CUDA-event ms per stage key."""

    def __init__(self, dev):
        self.dev = dev
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, key, obj, name):
        orig = getattr(obj, name)

        def timed(*a, **kw):
            if self.dev.type != "cuda":
                t0 = time.perf_counter()
                r = orig(*a, **kw)
                self.ms[key] += (time.perf_counter() - t0) * 1e3
                self.calls[key] += 1
                return r
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            r = orig(*a, **kw)
            e1.record()
            torch.cuda.synchronize(self.dev)
            self.ms[key] += e0.elapsed_time(e1)
            self.calls[key] += 1
            return r

        setattr(obj, name, timed)
        return orig

    def reset(self):
        self.ms.clear()
        self.calls.clear()


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = load_model(a.ckpt, device=a.device, dtype=dtype)
    dev = model.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t = Timer(dev)
    # top-level exclusive stages
    t.wrap("embed", model.embed, "rows")
    t.wrap("rms_norm", tr, "rms_norm")
    t.wrap("attention", model, "_attention")

    # _mlp split by tag: dense layers vs the shared-expert call inside _moe
    true_mlp = tr.Model._mlp

    def mlp_timed(self, x, tag):
        key = "mlp_shared" if "shared_expert" in tag else "mlp_dense"
        if dev.type != "cuda":
            t0 = time.perf_counter()
            r = true_mlp(self, x, tag)
            t.ms[key] += (time.perf_counter() - t0) * 1e3
            t.calls[key] += 1
            return r
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        r = true_mlp(self, x, tag)
        e1.record()
        torch.cuda.synchronize(dev)
        t.ms[key] += e0.elapsed_time(e1)
        t.calls[key] += 1
        return r

    tr.Model._mlp = mlp_timed
    t.wrap("moe", model, "_moe")

    # dequant sub-view: every QLinear.dense / QLinear.expert keyed by tensor
    name_of = {id(q): n for n, q in model.q.items()}
    orig_dense = tr.QLinear.dense
    orig_expert = tr.QLinear.expert

    def dense_timed(self, dtype_):
        key = "dq:" + name_of.get(id(self), "?")
        if dev.type != "cuda":
            t0 = time.perf_counter()
            r = orig_dense(self, dtype_)
            t.ms[key] += (time.perf_counter() - t0) * 1e3
            return r
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        r = orig_dense(self, dtype_)
        e1.record()
        torch.cuda.synchronize(dev)
        t.ms[key] += e0.elapsed_time(e1)
        return r

    def expert_timed(self, e, dtype_):
        key = "expert_dq"
        if dev.type != "cuda":
            t0 = time.perf_counter()
            r = orig_expert(self, e, dtype_)
            t.ms[key] += (time.perf_counter() - t0) * 1e3
            return r
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        r = orig_expert(self, e, dtype_)
        e1.record()
        torch.cuda.synchronize(dev)
        t.ms[key] += e0.elapsed_time(e1)
        return r

    tr.QLinear.dense = dense_timed
    tr.QLinear.expert = expert_timed

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, model.cfg.vocab, (1, PREFILL), generator=g).to(dev)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    # warmups
    for _ in range(WARM_RUNS):
        model.forward(ids, model.new_cache())
    # timed prefill runs
    t.reset()
    wall = []
    for _ in range(RUNS):
        sync()
        t0 = time.perf_counter()
        logits = model.forward(ids, model.new_cache())
        sync()
        wall.append(time.perf_counter() - t0)
    pf_ms = {k: v / RUNS for k, v in t.ms.items()}
    pf_calls = dict(t.calls)
    prefill_tok_s = PREFILL / (sum(wall) / len(wall))

    # decode
    kv = model.new_cache()
    tok = int(model.forward(ids, kv)[0, -1].argmax())
    for _ in range(DECODE_WARM):
        tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
    t.reset()
    sync()
    t0 = time.perf_counter()
    for _ in range(DECODE):
        tok = int(model.forward(torch.tensor([[tok]], device=dev), kv)[0, -1].argmax())
    sync()
    decode_wall = time.perf_counter() - t0
    dc_ms = {k: v / DECODE for k, v in t.ms.items()}
    dc_calls = dict(t.calls)
    decode_tok_s = DECODE / decode_wall

    out = {
        "ckpt": str(a.ckpt), "tag": a.tag, "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__, "dtype": a.dtype,
        "prefill512_tok_s": round(prefill_tok_s, 1),
        "decode32_tok_s": round(decode_tok_s, 2),
        "decode_wall_ms_per_step": round(decode_wall / DECODE * 1e3, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None,
        "prefill_stage_ms": {k: round(v, 3) for k, v in sorted(pf_ms.items(), key=lambda kv: -kv[1])},
        "prefill_stage_calls": pf_calls,
        "decode_stage_ms_per_step": {k: round(v, 3) for k, v in sorted(dc_ms.items(), key=lambda kv: -kv[1])},
        "decode_stage_calls": dc_calls,
        "logits_finite": bool(torch.isfinite(logits).all()),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
