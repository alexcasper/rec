# rec benchmark run definitions (cross-GPU comparable)

Shared spec for the ops_rec dusk ROCm arm and the AUP other-vendor reference
arm (env_aup, w14:p4). Keep all numbers comparable to these definitions.

## Metric definitions

- **prefill tok/s** = prefill_tokens / wall-clock seconds, where
  prefill_tokens = 512 and wall-clock = time from kernel start to
  device-synchronize completion, averaged over 3 runs after 3 warmup runs on
  the same process/weights.
- **decode tok/s** = generated_tokens / wall-clock seconds, single stream,
  greedy (temperature 0), after the same 512-token prefill; measured over 32
  generated tokens after 4 warmup decode steps; device-synchronize before
  starting and after ending the timing window.
- No batching, single request, no speculative decoding, no prefix-cache reuse
  within the timed window.

## Model configs

Both are synthetic random-weight checkpoints (seed 0) matching the Flint
architecture family — identical shapes on every arm:

| Config | Layers | hidden | heads/kv | head_dim | experts (top-k) | moe_inter | shared_inter | dense_inter | vocab | weights |
|--------|--------|--------|----------|----------|-----------------|-----------|--------------|-------------|-------|---------|
| tiny-2L | 2 | 128 | 4/2 | 64 | 8 (2) | 64 | 64 | 256 | 512 | fp32 (~1 MiB) |
| mid-8L | 8 | 2048 | 32/4 | 128 | 32 (2) | 512 | 1024 | 2048 | 150144 | bf16 (2.85 GiB) |

Common details: sliding_window=513 (alternating sliding/full layers in mid-8L:
sliding on even layers), rope_theta=10000, RMSNorm eps=1e-6, attention sinks
per query head (random ±0.01), sigmoid router + shared expert on MoE layers,
layer 0 dense. Weight layout: same in-memory pinned/resident weights for the
whole run; no host→device copies inside timed regions.

## Environment matrix to report

gpu_model, gpu_arch, driver/runtime stack + version, torch/llama.cpp version,
weight dtype, kv dtype, context used, host RAM, any HSA/arch overrides.

## Reference numbers (dusk, Radeon 890M gfx1150, 2026-11-17)

- Stack: NixOS, ROCm toolchain 7.2 (nix), torch 2.9.1+rocm6.4,
  HSA_OVERRIDE_GFX_VERSION=11.0.0, bf16 weights pinned on device.
- tiny-2L: prefill512 117,563 tok/s; decode32 1,068.9 tok/s; peak 0.04 GiB.
- mid-8L: prefill512 2,462.8 tok/s; decode32 37.2 tok/s; peak 3.31 GiB.
- Incumbent on the same GPU (flint llama.cpp, Vulkan/RADV, Q3_K_XL, 131k ctx):
  decode ≈ 28.5 tok/s (cumulative), prefill gauge ≈ 404 tok/s.

## Reference numbers (albion, Radeon AI PRO R9700 gfx1201 RDNA4, 2026-11-17)

- Stack: NixOS, HIP 7.2.53211 (nix toolchain), torch 2.10.0+rocm7.0
  (pip, rocm7.0 index) — **no HSA override needed**, RDNA4 natively supported.
  NixOS LD_LIBRARY_PATH fix same pattern as dusk (gcc-11.5.0-lib, zstd, zlib).
  SSH: `albion@albion` (user albion; dawn/uther do not work). Connections
  rate-limit after bursts — use ControlMaster multiplexing.
- Three GPUs visible: 0 = R9700 (gfx1201, 32 GB), 1 = MI50 (gfx906, 32 GB),
  2 = Raphael iGPU (gfx1103, 23 GiB GTT). Benchmarks target cuda:0 (R9700).
- tiny-2L: prefill512 132,474 tok/s; decode32 985.6 tok/s; peak 0.04 GiB.
- mid-8L: prefill512 **13,126.7 tok/s**; decode32 **58.7 tok/s**; peak 3.31 GiB.
  (5.3× dusk prefill; decode only 1.6× — eager per-op dispatch dominates
  decode at this model size, so the discrete-GPU bandwidth advantage is
  masked until the fused decode path exists.)

## Reference numbers (albion, Instinct MI50 gfx906 CDNA1, 2026-11-18)

- Stack: torch **2.9.1+rocm6.3** (pip, rocm6.3 index) — **the only wheel line
  that still ships gfx906 kernels** (rocm6.4 and rocm7.0 wheels dropped it;
  TheRock gfx906 nightlies expired; do NOT attempt to mix the rocm7.0 stack
  with /opt/rocm 7.2.3 libs — RTLD_NOW makes every ABI mismatch fatal).
  Device cuda:1 (select with CUDA_VISIBLE_DEVICES=1).
- tiny-2L: conformance PASS (2e-6); prefill512 85,737 tok/s; decode32
  878.8 tok/s.
- mid-8L: prefill512 **2,000.4 tok/s**; decode32 **12.2 tok/s**; peak 3.31 GiB.
  MI50 is the slowest AMD arm (CDNA1, pre-bf16-era); useful as an
  other-architecture reference, not a speed target.

## Reference numbers — rec-9na INT4 torch model layer (dusk, Radeon 890M gfx1150, 2026-11-18)

Different weight config from the rows above. Weights stay **INT4-affine
packed** on device (INT8 embedding) and are dequantized inside every matmul
by `rec_mlx.torch_rocm`, with no fused kernel. Routed experts only are
dequantized. bf16 activations and KV. Checkpoints come from
`python -m rec_mlx.synthetic --preset {tiny,mid}` (seed 0, seeded router).
Script: `engine/tests/torch_rocm_bench.py` (same prefill/decode protocol as
above). Stack: torch 2.9.1+rocm6.4, HIP 6.4.43484, HSA_OVERRIDE_GFX_VERSION=11.0.0.

| config | resident weights | prefill512 tok/s | decode32 tok/s | peak |
|---|---|---|---|---|
| tiny-2L INT4 | <0.01 GiB | 91,963.6 | 495.1 | 0.05 GiB |
| mid-8L INT4 | 1.03 GiB | 498.2 | 2.25 | 7.1 GiB |

Conformance on the same run: `torch_rocm_tiny_check.py` PASS (dequant
bit-exact, device fp32 drift 2.1e-6, bf16 teacher-forced with 1 near-tie flip).

This is a correctness reference path, **not a speed claim**. Mid decode is
bound by dequantizing the full 150144×2048 lm_head plus the dense
projections every step, which is also why peak memory is 7.1 GiB.
**Superseded 2026-11-19 by the re-baseline section below** (dequant fix:
3.69–3.71 tok/s, 3.0–3.1 GiB peak, bit-identical numerics). The first
version, which dequantized all 32 experts, measured 1.06 tok/s. Do not
compare these numbers with the dense-bf16 rows above (58.7 / 37.2 / 12.2) as
like-for-like. The fused dequant-matmul (rec-aip) and the megakernel
(rec-tan) are the speed paths. Albion (R9700 / MI50) was unreachable for this
run.

## Re-baseline: rec-9na torch layer after the dequant fix (dusk, Radeon 890M gfx1150, 2026-11-19, rec-aip step 1)

**Superseded by Re-baseline 2 below.** This was the gate after step 1. It
superseded the 2.25 tok/s row above.

**Change.** `rec_mlx.torch_rocm.dequant` no longer materialises full-size
int32 unpack temporaries or `repeat_interleave`d fp32 scale and bias arrays
on every call. That was the waste behind 75–81% of a decode step in the
rec-ebj profile, §4 (`docs/benchmarks/aip-prep-dusk-gfx1150-2026-11-18.md`).
The new code:
- reads the packed words as bytes (an exact little-endian nibble split);
- converts once to fp32 `[..., groups, 64]`;
- applies scale and bias in place by broadcast;
- casts once.

**Numerics are bit-identical**, because the arithmetic is the same fp32
multiply, then add, with no FMA:
- `engine/tests/test_torch_rocm_dequant.py`: 25 cases, 2/4/8-bit, stacked
  experts, sign-bit words, bf16/f32 scales.
- On the dusk GPU, the real mid-prod lm_head, q_proj and dense down_proj
  (bf16 and fp32), a routed expert and INT8 embedding rows are all
  bit-identical.

The rec-pp2 validation reference is therefore unchanged. Weights stay INT4
packed on device; the s2 layout is untouched.

**Dequant microbenchmark** (`engine/tests/torch_rocm_dequant_micro.py`, mid
lm_head 150144×2048 → bf16):

| implementation | ms / call | transient | note |
|---|---|---|---|
| original (int32 shift + repeat_interleave) | 227.2 | 4.59 GiB | |
| strided per-nibble writes (rejected) | 327.5 | 1.72 GiB | less memory but slower: 8 uncoalesced passes |
| **bytes + in-place broadcast (landed)** | **134.2** | **1.72 GiB** | |

**Recipe.** `engine/tests/torch_rocm_bench.py`:
- prefill512: 3 warmup + 3 timed;
- decode32 after 512: 4 warmup + 32 timed;
- short-ctx decode32: prompt [1, 5, 6, 7], 4 warmup + 32 timed, mean of 3;
- serial-516: 512 prefill + 4 decode end to end, 516 / mean of 3.

All windows are canonical and sync-bounded (A1: eager path). The run used
torch 2.9.1+rocm6.4, HIP 6.4.43484, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, bf16
activations and KV. Order was **ABBA ×2 per checkpoint**: 4 old + 4 new
runs, old = the committed pre-fix `torch_rocm.py`, same bench script. Before
every run the co-tenants were idle (flint :8080 and qwen :8091 `/slots`:
0 processing). Values are medians, with the min–max over 4 runs in brackets.
The raw log is `rec-aip-rebaseline-dusk-2026-11-19.log`.

| ckpt | code | prefill512 tok/s | decode32 tok/s | short-ctx decode32 tok/s | serial-516 tok/s | peak |
|---|---|---|---|---|---|---|
| `cpu-baseline-mid` (synthetic, 20214438) | before | 496.4 [494.7–497.8] | 2.25 [2.24–2.25] | 2.26 [2.25–2.26] | 183.9 [183.3–184.0] | 7.11 GiB |
| `cpu-baseline-mid` | **after** | **739.0** [737.4–740.5] | **3.71** [3.71–3.72] | **3.75** [3.75–3.75] | **292.0** [291.7–292.9] | **3.09 GiB** |
| `mid-prod` (production format, f494e788) | before | 508.5 [500.2–510.0] | 2.24 [2.24–2.24] | 2.25 [2.25–2.25] | 184.7 [184.6–184.9] | 7.02 GiB |
| `mid-prod` | **after** | **751.8** [731.8–753.3] | **3.69** [3.69–3.69] | **3.73** [3.72–3.73] | **292.9** [292.6–293.5] | **3.00 GiB** |

Ratio after / before: prefill ×1.48, decode32 ×1.65, short-ctx ×1.66,
serial-516 ×1.59, peak ×0.43, the same on both checkpoints. The "before" rows
reproduce the committed 9c87cf3 numbers (2.25 tok/s, 7.1 GiB).

**Caveats.**
- An earlier A/B attempt that same morning, with the rejected strided
  variant, measured the *old* code at 0.48 tok/s decode, 4.7× below this run,
  with idle co-tenants. The 890M's ±2× swings (rec-ebj) are real. Only this
  ABBA run, with its tight spreads, is the run of record; the aborted log is
  not committed.
- The projection in rec-ebj §5 (~6–8 tok/s) assumed steps 1–3. This is step 1
  only. The lm_head dequant still runs every step at about 134 ms, so it
  remains the largest single cost. Caching the bf16 lm_head (perf flag),
  fused dequant-matmul, and the expert-loop host sync (§5 items 2–3) are
  next, and belong to the kernel lane.
- This is the eager torch path (canonical windows); the persistent-kernel
  rows use the A1 differential. State the protocol when comparing.
- gfx1150 only; albion was not run.

## Re-baseline 2: lm_head cache + sync-free decode MoE (dusk, Radeon 890M gfx1150, 2026-11-19, rec-aip step 2)

**This supersedes the step-1 re-baseline above as the eager-torch gate.**

**Changes in `rec_mlx.torch_rocm`:**
1. **lm_head cache.** `QLinear.dense()` keeps its result for the lm_head
   (`load_model(cache_lm_head=True)`, the default). The key is dtype plus
   tensor identity, plus the in-place version for tensors that track one. The
   cached tensor *is* the `dense()` output, so it is bit-identical. It costs
   +0.57 GiB resident in bf16 (1.15 GiB in fp32) and removes about 134 ms
   per forward.
2. **Sync-free single-token MoE.** For a single token, the routed experts
   come from the device top-k, sorted ascending as `torch.unique` returns
   them, and are dequantized via `index_select`. This drops the per-MoE-layer
   `unique().tolist()` / `nonzero()` host syncs, 7 per decode step, with the
   same ops, shapes and accumulation order. Prefill (more than one token)
   keeps the synced path: a sync-free version would change the GEMM M
   dimension, and so possibly the reduction order. That path is flagged, not
   changed.

**Bit-identity.**
- CPU: the 25 dequant cases are unchanged. `test_torch_rocm_decode_paths.py`
  compares whole-model logits, new paths against old, with `torch.equal` at
  every step. It covers bf16 and fp32, production and synthetic formats, and
  a run past the 513 window; it also tests cache reuse and invalidation,
  models loaded under inference mode, and that decode never calls
  `torch.unique`.
- dusk GPU, mid-prod (`engine/tests/torch_rocm_step2_gpu_check.py`): the
  cached lm_head in bf16 and fp32, routed experts 0/17/31 via device and host
  ids, and INT8 embed rows are all identical to the original dequant. The
  whole model, prefill 512 plus 16 decode steps, has identical logits at
  every step between new and old paths.

**Bench.** `torch_rocm_bench.py`, same recipe as step 1, ABBA ×2 per
checkpoint, old = `56ea8b7`. Before every run a wait loop held the run while
any Gemini megakernel process or busy flint/qwen slot existed; none was seen.
The run used torch 2.9.1+rocm6.4. Values are medians [min–max of 4]; the raw
log, spot checks and sync A/B are in `rec-aip-step2-dusk-2026-11-19.log`.

| ckpt | code | prefill512 tok/s | decode32 tok/s | short-ctx decode32 tok/s | serial-516 tok/s | peak |
|---|---|---|---|---|---|---|
| `cpu-baseline-mid` | 56ea8b7 | 738.5 [719.7–743.7] | 3.71 [3.71–3.71] | 3.75 [3.75–3.75] | 292.2 [291.9–292.8] | 3.09 GiB |
| `cpu-baseline-mid` | **step 2** | **915.7** [915.3–918.8] | **7.31** [7.30–7.33] | **7.47** [7.47–7.49] | **467.9** [467.1–469.9] | **2.80 GiB** |
| `mid-prod` | 56ea8b7 | 753.5 [737.0–753.8] | 3.69 [3.62–3.69] | 3.72 [3.67–3.73] | 292.7 [287.7–293.2] | 3.00 GiB |
| `mid-prod` | **step 2** | **933.8** [930.1–934.7] | **7.25** [7.25–7.27] | **7.42** [7.40–7.42] | **472.2** [471.5–472.9] | **2.71 GiB** |

Ratio step 2 / step 1: decode32 ×1.97, short-ctx ×1.99, serial-516 ×1.61,
prefill512 ×1.24, peak ×0.90. From the original 9c87cf3 baseline (2.25 tok/s,
7.1 GiB), decode is ×3.2 and peak ×0.38.

**Attribution.** In `engine/tests/torch_rocm_sync_ab.py` (mid-prod, cache on
in both arms, ABBA ×3), the MoE sync removal alone moves decode32 from
7.255 to 7.267 tok/s and short-ctx from 7.390 to 7.404 tok/s. That is about
+0.2%, with overlapping ranges, so it is **negligible**. At about 137 ms per
step, the 7 small syncs are hidden behind GPU work, and the bench's own
per-step `argmax().item()` syncs anyway. Effectively **all of the step-2 gain
is the lm_head cache.** The sync-free path stays in because it is
bit-identical and free, and it will matter once steps are short (fused
kernels).

**Caveats.**
- The prefill512 and serial-516 gains assume a warm cache. The bench repeats
  prefill, so only the first warmup pays the one-time lm_head dequant
  (~134 ms); a cold first request still pays it once per process.
- Resident weights grow 1.03 → 1.60 GiB with the bf16 cache, yet peak still
  drops because the per-step 1.72 GiB lm_head dequant transient is gone.
- The dense projections and routed experts are still dequantized every step.
  Those, plus a fused dequant-matmul, are the kernel lane's.
- An earlier launch crashed on `_version` of inference tensors (the bench
  loads under `inference_mode`). It was fixed, now covered by a test, and
  produced no numbers.
- The aborted strided attempt in the step-1 caveat overlapped a Gemini bench
  that finished at 07:54 (`bench_aip_w16.json`), which is the likely cause
  of its 0.48 tok/s reading. The step-1 run of record started afterwards.

## Prefill MoE host syncs: quantified, left in place (dusk 890M, mid-prod, 2026-11-19, rec-aip step 3)

**Measurement.** `engine/tests/torch_rocm_prefill_sync_probe.py` ran on
mid-prod with the rec-ikf gate before every timed forward: `gpu_busy_percent`
< 5 on 3 consecutive 0.3 s samples. Rounds were interleaved (12 rounds) and
each forward was timed with CUDA events. The raw data is in
`rec-aip-step3-prefill-sync-dusk-2026-11-19.json`.

Stock prefill512 makes **164 host syncs** per forward: per MoE layer one
`unique().tolist()` plus one `nonzero()` per hit expert, with 149 experts hit
across the 7 MoE layers. An oracle replay of the identical ops and shapes,
with no syncs, is **6.4 ms faster** (paired median, 12 of 12 rounds
positive): 549.7 → 543.2 ms, i.e. **1.2% of prefill**. Consolidating to one
`bincount().tolist()` per layer (21 syncs) keeps the shapes, is verified
bit-identical on GPU, and recovers 97% of that, 6.0 ms. All variants are
`torch.equal` to stock.

**Why there is no sync-free, numerics-preserving prefill.** Each expert's
GEMM M dimension (tokens routed to it) exists only on the device after the
router runs. hipBLAS/rocBLAS launches take M, N and K as host arguments, so
the host must either learn M, which is a sync, or launch with a fixed M (all
512 tokens, or a capacity bound). A fixed M changes the GEMM problem shape,
and with it kernel and tiling selection and per-row reduction order, so bit-
identity cannot be proven.

**Decision.** `prefill_sync_free` is **not** added, and the one-sync variant
is not landed in the reference layer. Its ~1.1% gain is below the prefill512
run-to-run spread (about 3%, step-2 table) and is not worth a second code
path in the conformance reference. The design and its bit-identity check are
in the probe if a later lane wants it. Device-side expert dispatch (grouped
GEMM) is a numerics-changing design, and belongs to the fused-kernel lane
under its own conformance run.

## Engine end-to-end, `REC_BACKEND=rocm` (dusk, Radeon 890M gfx1150, 2026-11-18, rec-ju9)

**Env (user-space, separate venv; the older `~/rec-ve/.venv` was left untouched):**

```bash
cd ~/rec-ve && uv venv --python 3.12 .venv-e2e
VIRTUAL_ENV=~/rec-ve/.venv-e2e uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/rocm6.4 \
  "torch==2.9.1+rocm6.4" "mlx[cpu]==0.32.0" "mlx-lm==0.31.3" \
  "transformers==5.16.1" "xgrammar==0.2.1" "pytest==8.4.2"
# ~/rec-ve/e2e-env.sh (NixOS: pip wheels need libstdc++/zstd/zlib from the store)
export LD_LIBRARY_PATH=<gcc-15.3.0-lib>/lib:<zstd-1.5.7>/lib:<zlib-1.3.2>/lib
export HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=$HOME/rec-ve/engine/src
python -m rec_mlx.synthetic --preset tiny --out ckpt-e2e/tiny   # 3 s
python -m rec_mlx.synthetic --preset mid  --out ckpt-e2e/mid    # 16 s, 1.1 GiB
REC_TORCH_DEVICE=cuda python engine/tests/rocm_engine_e2e.py --ckpt ckpt-e2e/<tiny|mid>
```

The engine is run from source (`PYTHONPATH`), not pip-installed, so the
native hatch build is not needed. The tiny checkpoint built on dusk is
byte-identical (sha256 `75168d28…`) to the one built on dawn. The engine
suite in this venv with `REC_TORCH_DEVICE=cuda`: 53 passed, 1 skipped (the
skipped test asserts that a missing GPU raises an error).

**Protocol** (`engine/tests/rocm_engine_e2e.py`): 512-token seeded prompt,
32 greedy tokens through `Engine.generate_tokens`, one 4-token warmup request
per engine. Timings are the engine's own metrics, so they include sampling,
detokenize and the per-step host copy of the logits. That is not the
bare-model protocol above. Conformance: the rocm engine in fp32
(`REC_TORCH_DTYPE=fp32`) must be greedy-identical to the MLX model
teacher-forced. Each bf16 flip must be explained by precision, meaning its
margin is at most the measured |bf16 − fp32| logit perturbation at that step.

| ckpt | engine | prefill512 tok/s | decode32 tok/s | conformance |
|---|---|---|---|---|
| tiny-2L | cpu (MLX-CPU) | 2,443.2 | 1,027.4 | reference |
| tiny-2L | rocm fp32 | 78,335.4 | 418.0 | 32/32 tokens identical to cpu engine |
| tiny-2L | rocm bf16 | 80,526.0 | 384.2 | 0 flips |
| mid-8L | cpu (MLX-CPU) | 5.9 | 2.60 | reference |
| mid-8L | rocm fp32 | 481.1 | 2.05 | 32/32 tokens identical to cpu engine |
| mid-8L | rocm bf16 | 517.2 | 2.16 | 2 flips (margins 5.8e-3, 7.4e-4 vs perturbation 0.32, 0.43): explained |

Overall result: **PASS** on both checkpoints. GPU prefill is 88× the MLX-CPU
engine on mid. GPU **decode is slower than MLX-CPU on mid** (2.16 vs 2.60
tok/s), because the INT4 reference path dequantizes the full lm_head every
step (see the rec-9na rows). This is a functional end-to-end validation, not
a speed result. The first rocm engine pays about 5 s of HIP init in
`load_seconds`.

**Served path** (`python -m rec_mlx.cli serve flint --model-dir D --manifest
D/manifest.json --grammar none --port 18099 --metrics m.jsonl`, with
`REC_BACKEND=rocm REC_TORCH_DEVICE=cuda`). Use `--grammar none`, because the
flint grammar is Metal-only, and a non-default port, because 8080 is the
shared flint llama-server. A `/v1/chat/completions` request with 16 tokens
at temperature 0 returns 200 on tiny and mid. The server log reports engine
decode of 386.4 tok/s on tiny and 2.12 tok/s on mid. The tiny completion
text is identical to the dawn cpu-backend server.

The first served attempt returned 200 with an empty body and then died with
`HSA_STATUS_ERROR_EXCEPTION`. The synthetic tokenizer had placed
specials and chat tokens at ids ≥ vocab_size, so the chat template fed the GPU
an out-of-bounds embedding gather. That is fixed in `rec_mlx.synthetic`, and
Engine now rejects such tokenizers at load and such ids per request
(commit 92f9bc2). Checkpoints built before that commit have bad tokenizers
and must be rebuilt; the weights are unchanged.

Scripts: `engine/.build/torch_conformance.py` (conformance + tiny bench),
`engine/.build/mid_bench_gpu.py` (mid bench, generates weights with seed 0).
AUP arm: replicate the mid-8L shapes exactly; report both configs if compute
allows, mid-8L minimum.

## Reference numbers — incumbent llama.cpp arm on the same mid-8L shape (dusk, Radeon 890M gfx1150, 2026-11-20, flint-y02)

Support-seat measurement (ops_flint) so ops_rec comparisons are same-shape.
The synthetic production checkpoint `~/rec-ve/ckpt-e2e/mid` (seed 0,
mlx-affine-int4, INT8 embeddings) was dequantized with the rec-aip
bit-exact dequant path (`mx.dequantize`, g64) and packed as a flint-arch
GGUF by `convert_mid_gguf.py` (copies in `docs/benchmarks/` and the flint
repo `tools/`). llama.cpp build b4-a37cc46, `build-vulkan` (Vulkan/RADV,
same binary as the flint :8080 server), `-ngl 99 -c 1024 -fa on`, f16 KV
cache, single stream greedy (temp 0, seed 0).

Protocol: prefill512 = 3 warmup + 3 timed; decode32 = 32 greedy tokens
after the same 512-token prefill. Windows are llama.cpp's internal
sync-bounded timing (prompt-eval rate / eval rate). Deviation from the
canonical windows above: llama-cli's decode window covers all 32 generated
steps — there are no separate 4 warmup decode steps. Prompt is 512 BOS
special tokens (the synthetic vocab has no free text ids; per-step shapes
and hence work are identical to a 512-token prefill, values differ).
Before every run the cotenant gate held: flint :8080 and qwen :8091
`/slots` 0 processing, gpu_busy 3%. Raw logs: `flint-y02-bench-f16.log`,
`flint-y02-bench-q4.log` (runs on dusk in `~/flint-mid/`).

| arm | weights resident | prefill512 tok/s | decode32 tok/s |
|---|---|---|---|
| mid-8L llama.cpp F16 | 3.06 GiB | 4,655.3 [4,522.1–4,668.6] | 64.8 [64.3–65.2] |
| mid-8L llama.cpp Q4_K_M | 0.91 GiB | 5,650.0 [5,634.9–5,704.0] | 157.3 [157.1–159.5] |

Context, same shape: eager torch bf16 37.2 (dusk), 58.7 (albion R9700),
megakernel 120.78 (run of record). The llama.cpp F16 arm on the dusk iGPU
is above both eager arms; the Q4 arm's decode implies ~37.6 GB/s effective
active-weight bandwidth, matching the ~38 GB/s measured on the full Flint
model (flint-5yy notes), so the numbers sit in the expected band.

Caveats: weights here are dequant(INT4), not the bf16 seed-0 values the
torch arms generate (random-weight checkpoint; values differ within
quantization error, timing unaffected). No token-parity gate vs the rec
engine was run — greedy determinism only. The Q4_K_M re-quantization uses
llama.cpp's own format, not the production mlx-affine-int4 layout.
gfx1150 only; albion was not run.
