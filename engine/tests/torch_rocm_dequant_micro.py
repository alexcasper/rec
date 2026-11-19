"""rec-aip: microbenchmark dequant formulations on the real mid lm_head (dusk GPU).
All candidates must be bit-identical to the original; times are ms per call.

    python engine/tests/torch_rocm_dequant_micro.py CKPT_DIR ENGINE_DIR   (ENGINE_DIR contains src/)
"""
import sys
import time

import torch

sys.path.insert(0, sys.argv[2] + "/src")
import rec_mlx.torch_rocm as T  # noqa: E402

G = T.GROUP_SIZE


def original(packed, scales, biases, bits, dtype):
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=packed.device, dtype=torch.int32) * bits
    q = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    q = q.reshape(*packed.shape[:-1], -1).to(torch.float32)
    return (q * scales.float().repeat_interleave(G, -1) + biases.float().repeat_interleave(G, -1)).to(dtype)


def strided(packed, scales, biases, bits, dtype):
    per_word = 32 // bits
    K = packed.shape[-1] * per_word
    q = torch.empty(*packed.shape, per_word, device=packed.device, dtype=torch.float32)
    for j in range(per_word):
        q[..., j] = (packed >> (j * bits)) & ((1 << bits) - 1)
    q = q.view(*packed.shape[:-1], K // G, G)
    q.mul_(scales.float().unsqueeze(-1)).add_(biases.float().unsqueeze(-1))
    return q.view(*packed.shape[:-1], K).to(dtype)


def bytes_(packed, scales, biases, bits, dtype):
    b = packed.contiguous().view(torch.uint8)
    if bits == 4:
        q = torch.stack((b & 15, b >> 4), dim=-1)
    else:
        q = b
    K = packed.shape[-1] * (32 // bits)
    q = q.reshape(*packed.shape[:-1], K // G, G).to(torch.float32)
    q.mul_(scales.float().unsqueeze(-1)).add_(biases.float().unsqueeze(-1))
    return q.view(*packed.shape[:-1], K).to(dtype)


def bytes_bcast(packed, scales, biases, bits, dtype):
    """bytes + out-of-place broadcast mul/add (one fewer pass than mul_/add_?)."""
    b = packed.contiguous().view(torch.uint8)
    q = torch.stack((b & 15, b >> 4), dim=-1) if bits == 4 else b
    K = packed.shape[-1] * (32 // bits)
    q = q.reshape(*packed.shape[:-1], K // G, G).to(torch.float32)
    return (q * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).view(*packed.shape[:-1], K).to(dtype)


m = T.load_model(sys.argv[1], device="cuda", dtype=torch.bfloat16)
cands = {"original": original, "strided": strided, "bytes": bytes_, "bytes_bcast": bytes_bcast}
for name in ("lm_head", "model.layers.1.self_attn.q_proj"):
    q = m.q[name]
    ref = original(q.packed, q.scales, q.biases, q.bits, torch.bfloat16)
    for cn, fn in cands.items():
        out = fn(q.packed, q.scales, q.biases, q.bits, torch.bfloat16)
        same = torch.equal(out, ref)
        del out
        for _ in range(2):
            fn(q.packed, q.scales, q.biases, q.bits, torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            fn(q.packed, q.scales, q.biases, q.bits, torch.bfloat16)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        extra = (torch.cuda.max_memory_allocated() - base) / 2**30
        print(f"{name:34} {cn:12} {ms:8.2f} ms  transient {extra:5.2f} GiB  bit-identical={same}", flush=True)
    del ref
    q8 = m.embed
    ids = torch.arange(0, 150144, 37, device="cuda")
    r = original(q8.packed[ids], q8.scales[ids], q8.biases[ids], q8.bits, torch.bfloat16)
    print("embed int8 rows bytes bit-identical:",
          torch.equal(bytes_(q8.packed[ids], q8.scales[ids], q8.biases[ids], q8.bits, torch.bfloat16), r))
