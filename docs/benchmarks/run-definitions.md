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

Scripts: `engine/.build/torch_conformance.py` (conformance + tiny bench),
`engine/.build/mid_bench_gpu.py` (mid bench, generates weights with seed 0).
AUP arm: replicate the mid-8L shapes exactly; report both configs if compute
allows, mid-8L minimum.
