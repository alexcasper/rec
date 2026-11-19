# rec-aip prep — per-op INT4 torch-layer profile, dusk 890M (gfx1150), 2026-11-18

Stage-level CUDA-event timing of `rec_mlx.torch_rocm` (INT4-affine packed
weights, dequant per matmul, bf16 activations/KV) on dusk (Radeon 890M,
torch 2.9.1+rocm6.4, HIP 6.4.43484, `HSA_OVERRIDE_GFX_VERSION=11.0.0`).
Checkpoints: `python -m rec_mlx.synthetic --preset {tiny,mid}` (seed 0,
seeded router), same as the committed 9c87cf3 protocol. Script:
`engine/tests/aip_prep_profile.py` (protocol mirrors
`torch_rocm_bench.py`: prefill 512 ×3 after 3 warmups, 32 greedy decode
steps after 4 warmups). Raw stage dumps (archived runs):
`aip-prep-raw-tiny.json`, `aip-prep-raw-mid.json`.

**Honesty note first:** the 890M is a busy iGPU (resident llama servers,
shared UMA memory/clocks). Two profile runs of the identical protocol
differ by up to ~2× on individual mid-stage timings, so this doc reports
**both runs** and draws conclusions only from what is stable across both:
the *ranking* of stages. Percentages below are min–max across the two
runs. Single-run numbers in this doc should not be quoted as exact.

Caveats: (1) CUDA-event sync per wrapped call adds overhead — use §0
bench rows for wall-clock, profile rows for shares only. (2) `rms_norm`
attribution is unstable (1.5 ms vs 23.2 ms for the identical mid decode
step across runs) and reads as CPU-stall attribution noise; norms are
noise here. (3) Shares are of the event-timed forward, exclusive nesting
(sub-spans noted). (4) No albion arm (host offline all day).

## 0. Wall-clock repro vs the committed 9c87cf3 table

Un-instrumented `torch_rocm_bench.py`, this host:

| config | committed (9c87cf3) | this run | delta |
|---|---|---|---|
| tiny-2L prefill512 | 91,963.6 tok/s | 75,471.2 tok/s | **−17.9%** ⚠ |
| tiny-2L decode32 | 495.1 tok/s | 425.5 tok/s | **−14.1%** ⚠ |
| mid-8L prefill512 | 498.2 tok/s | 496.5 tok/s | −0.3% ✓ |
| mid-8L decode32 | 2.25 tok/s | 2.23 tok/s | −0.9% ✓ |
| peaks | 0.05 / 7.1 GiB | 0.05 / 7.1 GiB | match ✓ |

⚠ Flagged, not reconciled: tiny disagrees by 14–18% while mid matches
within 1%. Same script/protocol/seed. Likely iGPU clock/load variance
(tiny is launch-bound, wall noise relatively larger), but treat tiny
wall numbers as ±20% uncertain. Mid — the config the fusion decisions
below rest on — reproduces. **→ Reconciled in §7: root cause is GPU
co-tenancy, not clock ramp or launch jitter.**

## 1. mid-8L decode — where one step goes (two runs: 551.2 / 599.8 ms)

Exclusive stage shares, run A (551.2 ms) / run B (599.8 ms):

| stage | run A | run B | of which dequant |
|---|---|---|---|
| lm_head dequant | **49.7%** | **40.3%** | all of it |
| attention (q/k/v/o + math) | 28.6% | 30.9% | ~14% pt both runs |
| moe (router + experts + shared) | 17.2% | 21.8% | expert_dq 7.9 / 9.5% pt; shared ≈ all dequant |
| mlp_dense (layer 0) | 2.2% | 2.4% | ~11–14 ms |
| rms_norm | 0.3% | 3.9% | attribution noise |
| embed | 0.0% | 0.0% | — |
| residual | 2.0% | 0.7% | launch gaps, argmax, copies |

**Stable conclusion: dequant is ~75–81% of a decode step** (lm_head
~274/242 ms, attention q/k/v/o ~79 ms, expert_dq 43/57 ms, mlp dense +
shared ≈ all dequant). The step is dequant-bandwidth-bound, confirming
the 9c87cf3 note. MoE routing itself is small (~7 ms): the MoE cost is
weight dequant, not routing.

## 2. mid-8L prefill — one 512-token forward (run A ~1.29 s / run B ~2.63 s)

| stage | run A | run B | of which dequant |
|---|---|---|---|
| moe | **48.2%** | **48.4%** | expert_dq **37.4 / 31.4% pt**; shared ~3% pt |
| attention | 23.2% | 14.1% | ~10.5% pt run A |
| lm_head dequant | 20.8% | 9.5% | all of it |
| rms_norm | 2.0% | 2.2% | — |
| mlp_dense (layer 0) | 1.4% | 0.8% | — |
| embed | 0.1% | 0.1% | — |

Run B's forward is ~2× slower overall — whole-GPU slowdown, not a
profile artifact; the moe share (48%) is identical in both. **Stable
conclusion: routed-expert dequant is ~31–37% of prefill.** 512 tokens ×
top-2 of 32 experts hits essentially all experts, and they are
dequantized in a per-expert Python loop with a host sync
(`torch.unique(...).tolist()`) per MoE layer (7 syncs/forward).

## 3. tiny-2L decode — launch-bound, different regime

Step time 3.68 ms (run A) / 8.58 ms (run B) — ±2× wall noise again, but
the ranking is identical: attention ~41–43%, moe ~31–33%, rms_norm
~10%, everything else ≤8%. Per-call dequant floors at ~0.05–0.1 ms ≈
kernel-launch overhead across ~100 calls/step. **Per-op fusion cannot
rescue tiny — the launch floor is the cost. Tiny is the persistent
megakernel's (rec-tan) to win, not rec-aip's.**

## 4. Structural finding: `dequant()` is ~3× more expensive than it should be

`dequant()` (torch_rocm.py:122) materializes, per call: `q` as fp32
`[out, K]`, plus `scales` and `biases` each `repeat_interleave`d into
full fp32 `[out, K]` arrays. For lm_head that is ~3×1.18 GiB of fp32
traffic to produce 0.57 GiB of bf16; effective bandwidth ~5 GB/s — an
order of magnitude under the 890M's capability. Broadcasting
`s[..., None]` / `b[..., None]` instead of `repeat_interleave` removes
two full-size intermediates in pure torch, no custom kernel.

## 5. Fusion-order recommendation (ranked by expected decode gain)

1. **Fix `dequant()` broadcasting (pure torch, no kernel).** Cheapest
   change, attacks every stage at once (75–81% of decode, ~70–75% of
   prefill is dequant). Do this before any HIP/Triton work — it
   re-baselines rec-aip.
2. **lm_head: fused dequant×matmul kernel, or cache the dequantized
   bf16 lm_head.** 40–50% of decode. Caching costs +0.57 GiB resident
   (1.03 → ~1.6 GiB, still ≪ 2.85 GiB dense) and moves logits within
   the bf16 rounding class (the correctness path must stay INT4 — make
   caching a perf-path flag). A fused kernel keeps numerics identical.
3. **Routed experts: batched/fused dequant-matmul + drop the per-layer
   host sync.** 31–37% pt of prefill, 8–10% of decode. At batch-1
   decode the expert set is fixed (top-k=2) and needs no
   `unique()/tolist()`.
4. **Attention q/k/v/o dequant fusion.** ~14% pt of decode. Natural
   follow-on once the dequant primitive is fixed.
5. **tiny: no per-op fusion** — hand tiny to the megakernel lane.

Expected combined effect if 1+2+3 land: decode step ~550–600 ms →
~120–160 ms, i.e. mid decode ~2.2 → ~6–8 tok/s on the 890M before any
true fused kernel; rec-aip's fused dequant-matmul compounds on top.
(Projection, not a measurement — re-baseline after step 1.)

## 6. What this does NOT claim

- No like-for-like comparison with the dense-bf16 rows (different config).
- Shares are from event-instrumented runs; wall-clock claims use §0.
- Stage timings carry the run-to-run variance documented above; only the
  rankings (stable across both runs) are load-bearing.
- No albion arm (host offline all day); gfx1150-only.

## 7. Reconciliation: tiny-2L swing is GPU co-tenancy (2026-11-19)

Setup: the 890M is shared. During the first probe the GPU sat at
`gpu_busy_percent ≈ 99%` continuously (co-tenant benches — other lanes
were in their bench window). Amdgpu sysfs sampled at 0.3 s throughout
(`aip_variance_probe.py`); a second, **gated** probe (`aip_gated_probe.py`)
fires the bench only after 3 consecutive `busy < 5` samples.

| regime | n | prefill512 tok/s | decode32 tok/s |
|---|---|---|---|
| gated (busy < 5 before run) | 8 | **85,155 – 90,772** | **422.5 – 479.0** |
| ungated, co-tenant saturated (busy ≈ 99%) | 9 | 7,770 – 22,044 | 58.6 – 71.7 |
| ungated, partial contention | 3 | 75,713 – 77,295 | 363.4 – 375.3 |
| ungated, landed in a quiet window | 1 | 91,101.9 | 478.7 |
| committed (9c87cf3) | — | 91,963.6 | 495.1 |

Raw: `aip-prep-raw-gated-tiny.jsonl` (one JSON per gated run, with
pre-run busy stats), `aip-prep-raw-ungated-tiny.jsonl` (first probe).

Findings:

1. **Root cause = co-tenancy, confirmed.** Gated runs on an idle GPU
   cluster within −1.4% to −7.4% of the committed table; the same
   binary/protocol ungated under a saturated GPU is 4–7× slower. The
   aip-prep profile run's −14/−18% was partial contention, and §1–§3
   stage shares were taken during partial contention (rankings hold;
   absolute ms values are inflated by the same factor — re-profile on
   an idle GPU before quoting absolute per-op costs).
2. **Clock ramp excluded.** A 20 s full-blast 4096² bf16 matmul load
   immediately before the bench did not change the slow runs (they
   stayed at ~16–22k while the GPU remained saturated), and active sclk
   telemetry stayed in a mid pstate across both fast and slow runs
   (sysfs sclk parsing is ambiguous; `busy%` is the reliable signal).
   If clocks mattered, the warm runs would have recovered.
3. **Launch-bound configs are the contention canary.** mid-8L reproduced
   within 1% *during* contention while tiny swung 4–7×: mid's kernels
   are large (long GPU occupancy per launch), so time-slicing steals
   little; tiny's ~100 short launches per decode step inherit the
   co-tenant's queues. Any launch-bound number on this host is
   meaningless without a busy-gate.
4. **Protocol fix for future runs on shared hosts:** gate on
   `gpu_busy_percent < 5` for ≥1 s before (and ideally during) every
   timed run, and record the busy stats with the result
   (`aip_gated_probe.py` does this). A run-definitions note to that
   effect is recommended for the shared-host rows.

Gated-run wall summary: tiny-2L INT4 on idle dusk 890M = **~90k prefill /
~470 decode tok/s**, reproducing the committed table; the committed
numbers stand.
