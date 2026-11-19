"""(dusk GPU tool; usage: python tests/torch_rocm_step2_gpu_check.py CKPT ENGINE_DIR)
rec-aip step 2 GPU spot checks + decode-cost decomposition on dusk (mid-prod).

    python gpu_step2.py CKPT ENGINE_DIR
1. lm_head cache: cached tensor is reused and bit-identical to the original dequant (bf16, fp32).
2. expert_at (device id) == expert (host id) == original dequant, for a routed expert.
3. INT8 embed rows == original dequant.
4. Whole model, prefill 512 + 16 decode: new paths vs old paths, logits torch.equal per step.
5. Decode32-after-512 time with the two switches on/off (interleaved, 2 rounds).
"""
import sys
import time

import torch

sys.path.insert(0, sys.argv[2] + "/src")
sys.path.insert(0, sys.argv[2] + "/tests")
from test_torch_rocm_dequant import dequant_reference  # noqa: E402

import rec_mlx.torch_rocm as T  # noqa: E402

m = T.load_model(sys.argv[1], device="cuda", dtype=torch.bfloat16)
lm = m.q["lm_head"]
for dt in (torch.bfloat16, torch.float32):
    a = lm.dense(dt)
    same_obj = lm.dense(dt) is a
    print("lm_head cache", dt, "reused" if same_obj else "RECOMPUTED",
          "bit-identical" if torch.equal(a, dequant_reference(lm.packed, lm.scales, lm.biases, lm.bits, dt)) else "DIFFERENT",
          f"{lm.cached_nbytes() / 2**30:.3f} GiB", flush=True)
    del a
lm._cached, lm._cache_key = None, None
e = m.q["model.layers.3.mlp.switch_mlp.down_proj"]
for ex in (0, 17, 31):
    t = torch.tensor([ex], device="cuda")
    r = dequant_reference(e.packed[ex], e.scales[ex], e.biases[ex], e.bits, torch.bfloat16)
    print("expert", ex, torch.equal(e.expert_at(t, torch.bfloat16), r), torch.equal(e.expert(ex, torch.bfloat16), r))
ids = torch.tensor([0, 1, 5, 4242, 150143], device="cuda")
print("embed rows", torch.equal(m.embed.rows(ids, torch.bfloat16),
                                dequant_reference(m.embed.packed[ids], m.embed.scales[ids], m.embed.biases[ids],
                                                  m.embed.bits, torch.bfloat16)), flush=True)

g = torch.Generator().manual_seed(0)
prompt = torch.randint(3, m.cfg.vocab, (1, 512), generator=g).cuda()


def set_paths(cache, syncfree):
    lm.cache_dense = cache
    lm._cached, lm._cache_key = None, None
    m.moe_sync_free_decode = syncfree


@torch.inference_mode()
def trajectory(n):
    kv = m.new_cache()
    lg = m.forward(prompt, kv)
    out = [lg[0, -1].float().cpu()]
    tok = int(lg[0, -1].argmax())
    for _ in range(n):
        lg = m.forward(torch.tensor([[tok]], device="cuda"), kv)
        out.append(lg[0, -1].float().cpu())
        tok = int(lg[0, -1].argmax())
    return out


set_paths(True, True)
new = trajectory(16)
set_paths(False, False)
old = trajectory(16)
print("model prefill512+16 decode, new vs old paths:",
      "bit-identical" if all(torch.equal(a, b) for a, b in zip(new, old)) else
      f"DIFFERENT at steps {[i for i, (a, b) in enumerate(zip(new, old)) if not torch.equal(a, b)]}", flush=True)


@torch.inference_mode()
def decode32():
    kv = m.new_cache()
    tok = int(m.forward(prompt, kv)[0, -1].argmax())
    for _ in range(4):
        tok = int(m.forward(torch.tensor([[tok]], device="cuda"), kv)[0, -1].argmax())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(32):
        tok = int(m.forward(torch.tensor([[tok]], device="cuda"), kv)[0, -1].argmax())
    torch.cuda.synchronize()
    return 32 / (time.perf_counter() - t0)


cfgs = [(False, False), (True, False), (False, True), (True, True)]
res = {c: [] for c in cfgs}
for rnd in range(2):
    for c in (cfgs if rnd == 0 else cfgs[::-1]):
        set_paths(*c)
        res[c].append(decode32())
for c in cfgs:
    print(f"decode32 cache_lm_head={c[0]!s:5} moe_sync_free={c[1]!s:5}: "
          f"{sum(res[c]) / len(res[c]):.2f} tok/s  {[round(x, 2) for x in res[c]]}", flush=True)
