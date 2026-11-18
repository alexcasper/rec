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

## Reference numbers — rec-tan Phase 2b HIP Persistent Megakernel & Fused INT4 Matmul (dusk, Radeon 890M gfx1150, 2026-11-18)

Multi-workgroup persistent single-launch greedy decode megakernel
(`engine/native/megakernel_hip/flint_mid_megakernel.hip` +
`mid_mk_driver.py`), consuming the exact zero-copy layout in
`docs/rocm-model-layer.md` s2 & s4 (U8 INT4 packed nibbles, BF16 scales/biases
with element-unit expert strides, INT8 affine embeddings, head-major BF16 KV
with 513-slot ring on even sliding layers and linear growth on odd full
layers). Compiled with TheRock ROCm 10.0.0 (`hipcc -O2 --offload-arch=gfx1150`),
launched with `workers=16` workgroups × 256 threads (128 wave32s across 16 CUs)
synchronized via device-wide `sync_all`.

| Config / Kernel | Resident Bank | Decode32 (after 512 ctx) | Decode32 (short ctx) | End-to-End 516-step Serial | Conformance vs fp64 Reference |
|---|---|---|---|---|---|
| `mid-8L` INT4 persistent megakernel (`mid_generate`) | **0.944 GiB** | **99.00 tok/s** | **139.60 tok/s** | **124.08 tok/s** | **MATCH** (16/16 & 24/24 tokens exact, rel maxabs err ≤ 5.84e-3) |
| `tiny-2L` INT4 persistent megakernel (`tiny_generate`, `workers=4..8`) | <0.01 GiB | — | — | — | **11/11 MATCH** (seeds 0..4 × prompts 4,6 × 24 tok + legacy, rel err ≤ 9.12e-4) |

Standalone fused INT4-affine dequant-matmul (`flint_q4_gemv_kernel`, zero-copy U8+BF16 s/b, MSL period-4 pre-scale) on `dusk` `gfx1150`:

| Projection Shape (`[N, K]`) | Latency (ms) | Effective Bandwidth (GB/s) | Max Abs Error vs fp64 | Relative MaxAbs Error |
|---|---|---|---|---|
| `expert_gate` `[512, 2048]` | 0.010 ms | 60.69 GB/s | `1.788e-07` | `8.847e-08` |
| `q_proj` `[4096, 2048]` | 0.149 ms | 31.88 GB/s | `2.384e-07` | `8.931e-08` |
| `o_proj` `[2048, 4096]` | 0.166 ms | 28.47 GB/s | `2.384e-07` | `1.146e-07` |
| `lm_head` `[150144, 2048]` | 2.572 ms | **67.48 GB/s** | `2.384e-07` | `8.647e-08` |

Speedups on `dusk` `gfx1150` (`mid-8L` decode32 after 512-token prefill = **99.0 tok/s**, short-context decode32 = **139.6 tok/s**):
- **44.0×** faster than unfused torch INT4 reference (`2.25 tok/s`, and `0.944 GiB` vs `7.1 GiB` peak memory)
- **3.47×** faster than incumbent llama.cpp Vulkan (`28.5 tok/s`)
- **2.66×** faster than eager torch dense-bf16 floor (`37.2 tok/s`)
- **1.69×** faster on the 890M iGPU than the discrete R9700 eager torch dense-bf16 baseline (`58.7 tok/s`), because the single-launch persistent megakernel eliminates all per-op host dispatch and only touches the 2 routed experts + shared expert in packed INT4.

Scripts: `engine/.build/torch_conformance.py` (conformance + tiny bench),
`engine/.build/mid_bench_gpu.py` (mid bench, generates weights with seed 0).
AUP arm: replicate the mid-8L shapes exactly; report both configs if compute
allows, mid-8L minimum.
