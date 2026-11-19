# rec benchmark run definitions (cross-GPU comparable)

Shared spec for the ops_rec dusk ROCm arm and the AUP other-vendor reference
arm (env_aup, w14:p4). Keep all numbers comparable to these definitions.

## Metric definitions

- **Canonical prefill tok/s** = prefill_tokens / wall-clock seconds, where
  prefill_tokens = 512 and wall-clock = time from kernel start to
  device-synchronize completion, averaged over 3 runs after 3 warmup runs on
  the same process/weights.
- **Canonical decode tok/s** = generated_tokens / wall-clock seconds, single
  stream, greedy (temperature 0), after the same 512-token prefill; measured
  over 32 generated tokens after 4 warmup decode steps; device-synchronize
  before starting and after ending the timing window.
- No batching, single request, no speculative decoding, no prefix-cache reuse
  within the timed window.

### Persistent-Kernel Decode32 Protocol (Adopted Differential Protocol, R1–R8)

A single-launch persistent megakernel (`mid_generate` in
`engine/native/megakernel_hip/flint_mid_megakernel.hip`) executes the entire
token loop (`pos = 0 .. P+N-1`) inside one kernel launch and cannot perform a
mid-launch host synchronize. It runs the identical full forward — including
`lm_head` and device argmax — at every position (no prefill guard).

1. **Differential `decode32_after_512` (A1 + R4 + R5)**:
   - Launch A (`t516`): `prompt_len=512, n_out=4` (`516` positions: `512`
     prefill + `4` warmup-equivalent decode steps).
   - Launch B (`t548`): `prompt_len=512, n_out=36` (`548` positions: `512`
     prefill + `4` warmup-equivalent + `32` timed decode steps).
   - Both launches are bounded by `hipDeviceSynchronize()` inside
     `mid_session_run_ex`. Because greedy decode is deterministic, positions
     `0..515` (and all host `hipMemset` / prompt `hipMemcpy` work) are
     identical and cancel in `t548 - t516`, isolating the 32 forwards at
     positions `516..547`.
   - **Alternating ABBA pair order (R4)**: Execute 1 warmup launch at `516`
     positions, followed by **4 timed pairs in alternating `ABBA` order**
     (`(A,B)`, `(B,A)`, `(A,B)`, `(B,A)`). Alternating order cancels linear
     thermal and clock drift on shared/thermally-limited GPUs. Report each
     pair's `t516`, `t548`, `dt32 = t548 - t516`, per-pair `32 / dt32`, min/max
     spread across the 4 pairs, and active GPU co-tenants.
   - `decode32_after_512 = 32 / (mean(t548) - mean(t516))` over the 4 ABBA
     pairs. Always label rows produced by this method **`differential`**.
2. **Short-context differential `decode32_short_ctx` (A1 + R4)**:
   - Launch A (`t8`): `prompt_len=4, n_out=4` (`8` positions).
   - Launch B (`t40`): `prompt_len=4, n_out=36` (`40` positions).
   - Execute 2 warmup launches (`prompt_len=4, n_out=8`), then 4 timed pairs in
     alternating `ABBA` order. `decode32_short_ctx = 32 / (mean(t40) - mean(t8))`.
3. **End-to-end 516-position throughput (A2)**:
   - `end_to_end_516_pos_tok_s = 516 / mean(t516)`. Because the persistent
     kernel runs a serial per-token forward including the `150144×2048`
     `lm_head` at every prompt position, this metric approximates the serial
     decode rate and is **not** a canonical batch prefill measure. Do not
     compare it like-for-like with canonical `prefill512` or `decode32` rows.
4. **Pinned benchmark prompts (A3 + R1)**:
   - `prompt512` (512 `int32` tokens):
     `np.random.default_rng(0).integers(3, 150144, size=512, dtype=np.int32)`
     - **`int32` little-endian `sha256`**:
       `4090cfe16f28a7deee8eadde2edf21ba8eb06655b5ad0f61c46b1f05164b4814`
     - First 8 tokens: `[127716, 95637, 76745, 40509, 46220, 6154, 11299, 2484]`.
     - *Note (R1)*: This NumPy PCG64 sequence is **not** identical to
       `engine/tests/torch_rocm_bench.py`, which draws from
       `torch.Generator().manual_seed(0)` (`torch.randint(3, vocab)`). Because
       MoE routing depends on token values, record which RNG generated the
       prompt.
   - `prompt4` (short-context 4 `int32` tokens): `[1, 5, 6, 7]` (note: token
     `5` is `<|assistant|>` in the synthetic tokenizer; `int32` `sha256`:
     `e8a26640d83fcc702f9fb94350c62e3da6bee25ad44c60aa882b712c0803d16d`).
   - These benchmark prompts are distinct from the `rec-pp2` correctness cases
     (`short4`, `chat`, `wrap512` in
     `engine/validation/megakernel_mid/mid_mk_check.py`).
5. **Session KV reuse & row-validity conditions (A5 + R2 + R5)**:
   - `mid_session_run_ex` reuses the session's `d_kv_sliding` and `d_kv_full`
     buffers without zeroing them between launches. Because every launch
     restarts at `pos = 0` and writes slot `pos % 513` (sliding) or `pos`
     (full) behind `SYNC_ALL` before causal attention reads `t in [t_start, pos]`,
     stale KV from prior launches is never read.
   - **Mandatory row-validity conditions (all required for a valid row)**:
     1. **No barrier abort (`R2`, `R5`)**: `sync_all` spins up to `50,000,000`
        iterations (`flint_mid_megakernel.hip` L63); on timeout it sets device
        abort flag `control[2] = phase + 1` and returns early from the kernel.
        `mid_session_run_ex` **must** copy `d_control` back to host after
        `hipDeviceSynchronize()` and verify `control[2] == 0`. Any launch with
        `control[2] != 0` or nonzero HIP return code is an aborted launch and
        **invalidates the benchmark row**.
     2. **Trajectory prefix invariance (`R5`)**: In every differential pair,
        the 4 generated tokens of the baseline launch (`n_out=4`) must equal
        the first 4 generated tokens of the extended launch (`n_out=36`):
        `tok_a == tok_b[:4]`.
     3. **Bounds & positive delta**: `prompt_len + n_out <= max_total`
        (enforced in `mid_session_run_ex`), every output token ID in
        `[0, vocab)`, and `dt32 > 0` for every pair.
6. **Workers cap, occupancy clamping, and co-residency (A6 + R3 + R6)**:
   - **Compiled hard cap (`R3`)**: `workers <= 64` (`MID_MAX_WORKERS = 64`).
     `ws_f_maxval` allocates 64 `float` slots (`ws_f_scores + 64 .. + 128`,
     immediately before `ws_f_logits`); `workers > 64` would overwrite
     `ws_f_logits` during the parallel argmax reduction. Both
     `mid_session_run_ex` and `mid_mk_driver.py` reject `workers < 1` or
     `workers > 64` loudly.
   - **Co-residency & occupancy clamping (`R6`)**: Because `sync_all` is a
     device-wide spin barrier across all `workers` blocks, every block must be
     co-resident simultaneously. Before launching, query
     `hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, 256, 0)`
     and `hipGetDeviceProperties(&prop, 0)` (exposed via `mid_query_occupancy`).
     The maximum co-resident worker count is
     `min(64, prop.multiProcessorCount * blocks_per_cu)`. Launching with
     `workers > max_co_resident_workers` triggers `control[2] != 0` barrier
     abort.

| Arch | GPU | Physical CUs (`prop.multiProcessorCount`) | Wave Size | Max Blocks/CU (`256` threads) | Max Co-Resident `workers` | Status |
|---|---|---|---|---|---|---|
| `gfx1150` | dusk Radeon 890M (RDNA 3.5, 8 WGPs / 16 CUs) | 8 WGPs (16 CUs) | 32 | 2 | **16** (`8` at 1/WGP) | Perf measured + `rec-pp2` **6/6 PASS** (`fp32`-KV 3/3 $\le 3.65\text{e-}6$, `bf16`-KV 3/3 $\le 7.87\text{e-}4$ with 1 accepted router near-tie on `chat` L2@pos19 `12->7`, gap `4.10e-5`) |
| `gfx1201` | albion R9700 (RDNA 4, Navi 48) | 32 WGPs (64 CUs) | 32 | query on host | $\le 64$ (hard cap 64) | Host offline (`rec-109` / `rec-a4w`) |
| `gfx906` | albion MI50 (CDNA 1, Vega 20) | 60 CUs | 64 | query on host | $\le 60$ (hard cap 64) | Host offline; requires ROCm 6.3 toolchain with `gfx906` device libs |

7. **Conformance gates and adopted sign-off policy (A7 + R7 + R8)**:

| Artifact | Checker / Reference | Near-Tie Margin (`TIE_TOL`) | Logit Relative Error Gate (`LOGIT_RTOL`) | Role |
|---|---|---|---|---|
| `tiny` matrix (`tiny_mk_driver.py` + `tiny_mk_np_reference.py`) | `fp64` NumPy mirror, pinned to MLX by `test_tiny_mk_reference.py` | `5e-3` | `2e-2` (script); `1e-3` (`rec-109` target) | Phase 2a tiny-2L gate |
| `mid` driver smoke (`mid_mk_driver.py --verify`) | `fp64` NumPy mirror `forward_mid_np` | `5e-2` | `2e-2` | Legacy driver smoke check (not sign-off) |
| **`mid` validation target (`rec-pp2`, `mid_mk_check.py`)** | `rec_mlx.torch_rocm` in `fp32` compute, teacher-forced; pinned to MLX by `test_torch_rocm.py` (`1.6e-6`) + `test_mid_mk_harness.py` | **`5e-3`** | **Two-tier: `fp32`-KV $\le 10^{-4}$ / `bf16`-KV $\le 10^{-3}$** over `last_logits` and all 24 `step_logits` rows | **THE mid-8L sign-off gate** |

   - **Adopted sign-off policy (`R7`)**:
     - Production kernels sign off via an **`fp32`-intermediate validation
       build** that passes all 3 cases (`short4`, `chat`, `wrap512`) in **both
       KV tiers** (`fp32`-KV $\le 10^{-4}$, `bf16`-KV $\le 10^{-3}$).
     - A direct `bf16`-activation run is not a substitute for sign-off.
   - *(Note per `R8`: `tiny` tiering item 3 from the draft proposal is dropped;
     any future `tiny` two-tier gate requires dedicated fault-injection
     calibration at tiny scale.)*

---

## Model configs

Both are synthetic random-weight checkpoints (seed 0) matching the Flint
architecture family — identical shapes on every arm:

| Config | Layers | hidden | heads/kv | head_dim | experts (top-k) | moe_inter | shared_inter | dense_inter | vocab | weights |
|--------|--------|--------|----------|----------|-----------------|-----------|--------------|-------------|-------|---------|
| tiny-2L | 2 | 128 | 4/2 | 64 | 8 (2) | 64 | 64 | 256 | 512 | fp32 (~1 MiB) |
| mid-8L | 8 | 2048 | 32/4 | 128 | 32 (2) | 512 | 1024 | 2048 | 150144 | bf16 (2.85 GiB) / INT4 packed (0.944 GiB) |

Common details: sliding_window=513 (alternating sliding/full layers in mid-8L:
sliding on even layers), rope_theta=10000, RMSNorm eps=1e-6, attention sinks
per query head (random ±0.01), sigmoid router + shared expert on MoE layers,
layer 0 dense. Weight layout: same in-memory pinned/resident weights for the
whole run; no host→device copies inside timed regions.

## Environment matrix to report

gpu_model, gpu_arch, CU/WGP count, wave size, workers, threads/workgroup,
occupancy clamp, driver/runtime stack + version, full build command + env
sourcing + runtime `LD_LIBRARY_PATH`, `HSA_OVERRIDE_GFX_VERSION` status,
python/numpy/torch versions, weight dtype, kv dtype, build tier, checkpoint
generator + `sha256`, prompt `sha256`, protocol (canonical vs differential
ABBA), active co-tenants, resident bank GiB, and session peak GiB.

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
Script: `engine/tests/torch_rocm_bench.py` (same canonical prefill/decode
protocol as above). Stack: torch 2.9.1+rocm6.4, HIP 6.4.43484,
HSA_OVERRIDE_GFX_VERSION=11.0.0.

| config | resident weights | prefill512 tok/s | decode32 tok/s | peak |
|---|---|---|---|---|
| tiny-2L INT4 | <0.01 GiB | 91,963.6 | 495.1 | 0.05 GiB |
| mid-8L INT4 | 1.03 GiB | 498.2 | 2.25 | 7.1 GiB |

Conformance on the same run: `torch_rocm_tiny_check.py` PASS (dequant
bit-exact, device fp32 drift 2.1e-6, bf16 teacher-forced with 1 near-tie flip).

This is a correctness reference path, **not a speed claim**. Mid decode is
bound by dequantizing the full 150144×2048 lm_head plus the dense
projections every step, which is also why peak memory is 7.1 GiB.
**Superseded 2026-11-19 by the torch dequant re-baseline (rec-9na / rec-aip
step 1, rec/claude-correctness `56ea8b7`): decode32 3.69–3.71 tok/s,
short-ctx 3.73–3.75, serial-516 292.0–292.9, peak 3.0–3.1 GiB, bit-identical
numerics. The 2.25 tok/s / 7.1 GiB row above is the pre-fix historical
reference.** The first
version, which dequantized all 32 experts, measured 1.06 tok/s. Do not
compare these numbers with the dense-bf16 rows above (58.7 / 37.2 / 12.2) as
like-for-like. The fused dequant-matmul (rec-aip) and the megakernel
(rec-tan) are the speed paths. Albion (R9700 / MI50) was unreachable for this
run.

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

## Reference numbers — rec-tan Phase 2b HIP Persistent Megakernel (dusk, Radeon 890M gfx1150, 2026-11-19, Corrected R1–R8 Protocol)

### Environment & Reproducibility Matrix (A8)
- **Host / GPU**: `dusk` — AMD Ryzen AI 9 HX 370 w/ Radeon 890M (`gfx1150`, RDNA 3.5, 8 WGPs / 16 CUs, `wave32`, 46 GiB host RAM shared GTT).
- **Toolchain & Runtime**: TheRock ROCm 10.0.0 (`~/opt/rocm-10.0.0`), sourced via `source ~/opt/rocm-env.sh` (`PATH=$HOME/opt/rocm-wrappers:$HOME/opt/rocm/bin:$PATH`, `LD_LIBRARY_PATH=$HOME/opt/rocm/lib:$HOME/opt/rocm-hostlibs`, resolving `libamdhip64.so.7` and `gcc-15.3.0-lib/libstdc++.so.6`). **No `HSA_OVERRIDE_GFX_VERSION`** (TheRock targets `gfx1150` natively).
- **Exact Build Command** (output `flint_mid.so` next to `mid_mk_driver.py` in `engine/native/megakernel_hip/`):
  ```bash
  source ~/opt/rocm-env.sh
  cd engine/native/megakernel_hip
  hipcc -shared -fPIC -O2 -Wall --offload-arch=gfx1150 flint_mid_megakernel.hip -o flint_mid.so
  ```
- **Driver & Python Env**: Torch-free driver process (`~/rec-ve/.venv/bin/python`, Python 3.12, NumPy 2.5.5):
  ```bash
  ~/rec-ve/.venv/bin/python mid_mk_driver.py --bench --ckpt ~/rec-ve/ckpt-pp2/mid-prod --workers=16 --mode=0
  ```
- **Checkpoint Pinned**: `~/rec-ve/ckpt-pp2/mid-prod`, generated via `python -m rec_mlx.synthetic --preset mid --format production` (U8 INT4 packed weights, INT8 embedding, BF16 scales/biases/norms/router/sinks, no `expert_bias`).
  - `model-00001.safetensors` `sha256`: `f494e7885d10bdedf5d885faef826a27fdbe4bfb1fb11781fac12c7cabbd92ad`
- **Build Tier Measured**: `mode=0` (`bf16-production`, `mid_generate`, `kv_dtype=bf16`).
- **Occupancy Query (`mid_query_occupancy`)**: `prop.multiProcessorCount = 8` (8 WGPs = 16 CUs on `gfx1150`), `max_blocks_per_cu = 2` (at 256 threads/block), **`max_co_resident_workers = 16`**, `wave_size = 32`, `compiled_hard_cap = 64`.
- **Active Co-Tenants on `dusk` iGPU during run (0.0% CPU / idle)**:
  1. `PID 2269`: `llama-server` Vulkan (`flint-preview-UD-Q3_K_XL.gguf`, `:8080`, `-c 131072 -ngl 99`)
  2. `PID 2300`: `llama-server` Vulkan (`google_gemma-4-E4B-it-Q4_K_M.gguf`, `:8090`, `-c 8192 -ngl 99`)
  3. `PID 2301`: `llama-server` Vulkan (`Qwen_Qwen3.5-4B-Q4_K_M.gguf`, `:8091`, `-c 32768 -ngl 99`)
- **Memory**: Resident weight bank = **`0.944 GiB`**; megakernel session peak (bank + sliding KV `0.0039 GiB` + full KV@600 `0.0046 GiB` + workspace/logits `0.0013 GiB`) = **`0.954 GiB`**.

### Decode32 Differential ABBA Results (`ckpt: f494e788…`, 4 ABBA pairs)

| Requested `workers` | Co-Resident Valid? | `control[2]` Abort Flag | `prefix4_match` (8/8 pairs) | Differential `decode32_after_512` (`tok/s`) [4-pair min..max] | Differential `decode32_short_ctx` (`tok/s`) [4-pair min..max] | End-to-End 516-pos (`512` prefill + `4` decode) (`tok/s`) | Mean `t516` / `t548` / `dt32` (`s`) |
|---|---|---|---|---|---|---|---|
| **`64`** (start cap) | **INVALID (Rejected)** | **`control[2] = 2`** (abort at phase 1, arrived=2 after 8.87s spin timeout) | — | — (exceeds `max_co_resident_workers = 16`) | — | — | — |
| **`16`** (`max_co_resident_workers`, 2 blocks/WGP = 1/CU) | **VALID (PASS)** | **`0`** (all 19 launches) | **`True`** (4/4 `ctx512` + 4/4 `short`) | **`111.67 tok/s`** [`105.34 .. 118.83`] | **`157.01 tok/s`** [`156.76 .. 157.34`] | **`131.67 tok/s`** | `3.9190s` / `4.2056s` / `0.2866s` |
| **`8`** (`prop.multiProcessorCount`, 1 block/WGP) | **VALID (PASS)** | **`0`** (all 19 launches) | **`True`** (4/4 `ctx512` + 4/4 `short`) | **`92.99 tok/s`** [`92.74 .. 93.33`] | **`122.05 tok/s`** [`121.96 .. 122.11`] | **`106.54 tok/s`** | `4.8434s` / `5.1875s` / `0.3441s` |

#### Per-Pair ABBA Breakdown (`workers = 16` and `workers = 8`)

- **`workers = 16` (`ctx512`: `prompt512` `sha256: 4090cfe1…`, `n_out=4` vs `36`)**:
  - Pair 0 (`AB`): `t516 = 3.89843s`, `t548 = 4.18575s`, `dt32 = 0.28732s` -> **`111.37 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 1 (`BA`): `t516 = 3.91490s`, `t548 = 4.18419s`, `dt32 = 0.26929s` -> **`118.83 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 2 (`AB`): `t516 = 3.91974s`, `t548 = 4.22351s`, `dt32 = 0.30377s` -> **`105.34 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 3 (`BA`): `t516 = 3.94293s`, `t548 = 4.22882s`, `dt32 = 0.28589s` -> **`111.93 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
- **`workers = 16` (`short_ctx`: `prompt4 = [1, 5, 6, 7]`, `n_out=4` vs `36`)**:
  - Pair 0 (`AB`): `t8 = 0.04978s`, `t40 = 0.25351s`, `dt32 = 0.20373s` -> **`157.07 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 1 (`BA`): `t8 = 0.05012s`, `t40 = 0.25425s`, `dt32 = 0.20413s` -> **`156.76 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 2 (`AB`): `t8 = 0.04968s`, `t40 = 0.25306s`, `dt32 = 0.20338s` -> **`157.34 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 3 (`BA`): `t8 = 0.05007s`, `t40 = 0.25408s`, `dt32 = 0.20401s` -> **`156.86 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
- **`workers = 8` (`ctx512`: `prompt512` `sha256: 4090cfe1…`, `n_out=4` vs `36`)**:
  - Pair 0 (`AB`): `t516 = 4.84223s`, `t548 = 5.18727s`, `dt32 = 0.34504s` -> **`92.74 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 1 (`BA`): `t516 = 4.84254s`, `t548 = 5.18657s`, `dt32 = 0.34402s` -> **`93.02 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 2 (`AB`): `t516 = 4.84325s`, `t548 = 5.18773s`, `dt32 = 0.34448s` -> **`92.89 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)
  - Pair 3 (`BA`): `t516 = 4.84552s`, `t548 = 5.18840s`, `dt32 = 0.34288s` -> **`93.33 tok/s`** (`ctrl_a=0, ctrl_b=0, prefix4=True`)

### Cross-Config Comparisons & Memory Labeling (A10)
- **vs. Unfused Torch INT4 reference (`rec-9na`, same host `dusk` `gfx1150`, same mid-8L INT4 architecture, different prompt RNG [Torch vs NumPy] and differential vs canonical protocol)**:
  - Throughput (current, post-`flint_q4_gemv_op` integration `d1f6208`): **`120.78 tok/s`** (`workers=16`) vs **`3.71 tok/s`** re-baselined unfused torch INT4 (`56ea8b7`; `3.69 tok/s` on the same mid-prod ckpt) = **`32.6×`** (**`32.7×`** same-ckpt). Historical pre-integration: `111.67` / `92.99 tok/s` (`workers=16`/`8`) vs the then-baseline `2.25 tok/s` = `49.6×` / `41.3×` — superseded. `workers=8` post-integration not yet measured.
  - Memory (resident-vs-resident and peak-vs-peak):
    - **Resident weights** (same mid-prod ckpt `f494e788`): `0.944 GiB` (megakernel) vs `0.943 GiB` (torch INT4) ≈ **`1.00×`** (identical packed weights). The former `1.09×` compared against cpu-baseline-mid's `1.03 GiB` — cross-checkpoint, superseded.
    - **Peak allocated GPU memory**: `0.954 GiB` (megakernel session peak) vs **`3.00 GiB`** (torch INT4 mid-prod peak after the dequant fix, `56ea8b7`; `3.09 GiB` on cpu-baseline-mid) = **`3.14×` lower peak** (`3.24×` vs cpu-baseline-mid). The `7.1 GiB` / `7.44×` figure was pre-fix — superseded.
- **vs. Eager Torch Dense-BF16 floor (same host `dusk` `gfx1150`, cross-config: INT4 persistent fused megakernel vs dense BF16 eager torch)**:
  - Throughput: `120.78 tok/s` (post-integration `d1f6208`) vs `37.2 tok/s` = **`3.25×` faster** (pre-integration `111.67 tok/s` = `3.00×`; `2.50×` at `workers=8` — historical, `workers=8` post-integration not yet measured).
  - Memory: `0.944 GiB` resident / `0.954 GiB` peak vs `2.85 GiB` resident / `3.31 GiB` peak.
- **vs. Eager Torch Dense-BF16 on discrete R9700 (`albion` `gfx1201`, cross-GPU AND cross-path: fused INT4 megakernel on 890M iGPU vs eager dense-BF16 torch on discrete RDNA4 dGPU)**:
  - Throughput: `120.78 tok/s` (post-integration `d1f6208`) vs `58.7 tok/s` = **`2.06×` faster** (pre-integration `111.67 tok/s` = `1.90×`; `1.58×` at `workers=8` — historical, `workers=8` post-integration not yet measured), because the single-launch megakernel eliminates per-op host launch overhead and reads only 2 routed experts + 1 shared expert in INT4.
- **vs. Incumbent `llama.cpp` Vulkan (`28.5 tok/s` on `dusk` `:8080`, NOT like-for-like — cross-model & cross-context caveat)**:
  - `120.78 / 28.5 = 4.24×` (post-integration `d1f6208`; pre-integration `111.67 / 28.5 = 3.92×`, `3.26×` at `workers=8` — historical, `workers=8` post-integration not yet measured), **caveat**: compares synthetic **mid-8L INT4 at 512-token context** against the real **Flint 49-layer `Q3_K_XL` GGUF at 131k context** cumulative `/metrics` decode rate. Must not be read as a like-for-like 49L comparison.

### Standalone Fused INT4-Affine Dequant-Matmul (`flint_q4_gemv_kernel`, A9) & Integrated `flint_q4_gemv_op` Decode Delta (`rec-aip`)
- **Method**: `flint_q4_gemv_host` wraps `iters=50` back-to-back `flint_q4_gemv_device` launches in `hipEventRecord` after `warmup=10` launches (`per_iter_ms = total_ms / 50`). Effective bandwidth formula counts weight+scale+bias+input reads plus fp32 output writes:
  `GB/s = (N*(K/2) + 2*N*(K/64)*2 + K*2 + N*4) / (per_iter_ms * 1e6)`.
- **Designated Run of Record (A9)**: The commit-adjacent `2026-11-18T23:36:35Z` run on `dusk` `gfx1150` is designated as the run of record for `c264e52`, with the prior `2026-11-18T23:35:01Z` compile+run recorded alongside it for run-to-run spread. Committed raw evidence logs:
  - Run of record (`23:36:35Z`): `docs/benchmarks/q4-hunt/step254-comment-run-2336Z.txt` (`sha256: a34911ab1c8e2a8a07e722846c5261b6b6610cfb610a3af9ddfa0de6dd0fc1c4`)
  - Prior compile+run (`23:35:01Z`): `docs/benchmarks/q4-hunt/step246-docs-run-2335Z.txt` (`sha256: 52de66b393d9d3c1adbf01485a8b320140bef484a5848175d6ac7a0d2bdeb92c`)

| Projection Shape (`[N, K]`) | Run-of-Record Latency (`ms`) | Run-of-Record Effective BW (`GB/s`) | Prior Run (`23:35Z`) Latency / BW | Max Abs Error vs `fp64` | Relative MaxAbs Error |
|---|---|---|---|---|---|
| `expert_gate` `[512, 2048]` | **`0.010 ms`** | **`61.23 GB/s`** | `0.010 ms` / `60.69 GB/s` | `1.788e-07` | `8.847e-08` |
| `q_proj` `[4096, 2048]` | **`0.139 ms`** | **`34.04 GB/s`** | `0.149 ms` / `31.88 GB/s` | `2.384e-07` | `8.931e-08` |
| `o_proj` `[2048, 4096]` | **`0.138 ms`** | **`34.25 GB/s`** | `0.166 ms` / `28.47 GB/s` | `2.384e-07` | `1.146e-07` |
| `lm_head` `[150144, 2048]` | **`2.264 ms`** | **`76.67 GB/s`** | `2.572 ms` / `67.48 GB/s` | `2.384e-07` | `8.647e-08` |

- **Integration into Mid-8L Megakernel Decode Path (`flint_q4_gemv_op<2048>`, `rec-aip`)**:
  - Unified `flint_q4_gemv_op<K>` across `flint_q4_gemv_kernel` and `mid_generate_impl`, consuming the zero-copy layout in `docs/rocm-model-layer.md` §2 (`U8 [.., out, K/2]`, `BF16` scales/biases `[.., out, K/64]`). For `K == 2048` (`lm_head` `[150144, 2048]`, which accounts for 40–50% of a mid decode step per `rec-ebj` profiling, plus `K=2048` layer projections), each lane preloads its 4×16 pre-scaled input slices (`v0..v3`) and 4 group sums (`sum0..sum3`) into registers once before the row loop and reads packed weights via 32-bit `uint32_t` loads in `dot16`.
  - **Before vs. After Decode Throughput (`workers = 16`, 4-pair ABBA differential recipe on `dusk` `gfx1150`, `ckpt: f494e788…`; JSON: `docs/benchmarks/bench_mid_prod_aip_q4_dusk_gfx1150.json`)**:
    - `decode32_after_512` (differential): **`111.67 tok/s` $\to$ `120.78 tok/s`** (**`+8.16%`**, `+9.11 tok/s`; 4-pair min..max `117.08 .. 123.49 tok/s`, `t516_mean = 3.5655s`, `t548_mean = 3.8304s`, `dt32_mean = 0.2649s` vs `0.2866s` before; **`32.7×`** vs the re-baselined unfused torch INT4 baseline **`3.69 tok/s`** decode32 on the same mid-prod ckpt `f494e788` (rec/claude-correctness `56ea8b7`, 2026-11-19 ABBA run of record; **`32.6×`** vs the `3.71 tok/s` cpu-baseline-mid headline). The former `2.25 tok/s` baseline (`53.7×`) is superseded. Protocol caveat unchanged: megakernel differential vs torch canonical sync-bounded windows).
    - `decode32_short_ctx` (differential): **`157.01 tok/s` $\to$ `175.67 tok/s`** (**`+11.88%`**, `+18.66 tok/s`; 4-pair min..max `174.17 .. 176.99 tok/s`, `dt_short_mean = 0.1822s` vs `0.2038s` before).
    - `end_to_end_516_pos_tok_s` (`512` prefill + `4` decode): **`131.67 tok/s` $\to$ `144.72 tok/s`** (**`+9.91%`**).
  - **Conformance (`rec-pp2` under `00c74a6` `all-declared` default policy)**: **6/6 PASS** (`exit 0`, both with `--emit-experts` on the `all-declared` `step_experts` path and without `--emit-experts` on the search fallback; see `engine/validation/megakernel_mid/results-rec-aip-matrix-all-declared.txt`).

Scripts: `engine/.build/torch_conformance.py` (conformance + tiny bench),
`engine/.build/mid_bench_gpu.py` (mid bench, generates weights with seed 0),
`engine/native/megakernel_hip/mid_mk_driver.py` (megakernel differential ABBA bench + `rec-mk-out/1` conformance emitter).
AUP arm: replicate the mid-8L shapes and pinned `prompt512` (`sha256: 4090cfe1…`) exactly; report both configs if compute allows, mid-8L minimum.
