# rec-9t8 on dusk (gfx1150, mid-8L): megakernel fused-dequant effective bandwidth

Go: coordinator 2026-11-21 ("go ahead with the fused dequant"). Checkpoint of record: RecursiveSI/flint-int4
(INT4 affine g64), so every lever keeps the s1/s2 layout. dusk rules: solo, launches <= 0.25 s (short4/chat
only, no wrap512/ctx512), busy < 5 before every round, flint :8080 / E4B :8090 logged per line and never
touched, llama-qwen disabled. R9700 full-Flint rows are albion-gated (offline).

Harness: the rec-6ob s4/s5 scripts (`run5.sh` = build + resources + G0 gate7 + lmh_dbg histograms + per-op
timing; `run5b.sh` = G1 suite byte identity vs 2fd83cb + ABBA decode32 short-ctx x8), plus `gtail.py` (generic-K
shapes with n % 4 != 0 at grids 0/1/3/16). Phase profile: `mk_prof.py` on an MK_PROFILE build
(`../6ob-dusk-2026-11-20/mk_profile.patch`).

## 0. Where the bandwidth goes (vcgen 39629e7, mk_prof, us/token; bytes = packed + bf16 s/b)

| phase | us | MB | GB/s |
|---|---|---|---|
| lm_head (150144 x 2048) | 2226 | 173 | 78 (ceiling ~82) |
| QKV (K=2048) | 743 | 47 | 63 |
| O-proj (K=4096) + residual | 898 | 38 | 42 |
| MLP (dense L0 + 7 MoE, 88 barriers) | 1305 | 57 | 44 |
| token | 5384 | ~315 | 59 |

At the ceiling the floor is ~3.8 ms/token (~260 tok/s). Losses are in the small-n phases, not lm_head.

## 1. Lever 1: batched MoE phases (b1)

Before: every routed expert ran gate/up -> SYNC -> act -> SYNC -> down -> SYNC, then the shared expert
did the same, then the combine: 3*TOPK + 4 grid barriers per MoE layer (mid 10; full Flint 22 x 47 layers).
After: all TOPK + shared gate/up in one phase, one act phase, all downs in one phase, then the combine:
4 barriers. Per-row gemvs and the act formula are unchanged; outputs go to per-expert slots of the existing
ws_gate / ws_up / ws_f_act (TOPK*512 + 1024 <= FFN_MAX on both configs; static_assert).

- G1 6/6 byte-identical to 2fd83cb. Barriers/token 140 -> 98.
- ABBA vs vcgen: short4 187.88 -> 193.03 (1.027x), chat 187.49 -> 191.96 (1.024x). Tokens identical.
- Profile: MLP 1305 -> 1162 us.
- Full Flint (TOPK 6) removes 18 barriers per MoE layer instead of 6; that effect is unmeasured until albion.

## 2. Lever 2: software-pipelined generic path (b2 -> b4)

The VCgen generic path (K != 2048) paid two dependent round trips per (quad, block): staged weights, then
the scattered bf16 scale/bias loads. b4 turns the full quads into one stream of (quad, block) stages. Stage
t+1's weights (2 x b128 per lane), 4+4 scale/bias bits and 16 raw x floats are issued before stage t's
compute. Weights land in the other half of a per-warp 2 KiB LDS double buffer, behind a wave fence pair
that guards only that half. Same words, same scales, the pre-scale expression copied from load16_f, same
block order -> byte-identical by construction. One LDS pool (`q4_gen_pool`) is shared by every generic-K
instantiation: mid_generate LDS 32.9 -> 24.7 KiB (s5 showed growth past ~33 KiB breaks persistent-WG
co-residency on dusk).

| build | change | VGPR K=4096 / 1024 / 512 | scratch | o_proj | expert_down |
|---|---|---|---|---|---|
| b2 | pipeline, `unroll 2` | 248 / - / - | 36 B in-loop | 1.237x | 0.964x |
| b3 | `unroll 1` | 215 / 214 / 213 | 0 | 1.293x | 0.913x |
| b4 | b3 + K=512 keeps VCgen single stage (stage = whole quad) | 215 / 214 / 115 | 0 | **1.315x** | **1.151x** |

(standalone grid16 vs 2fd83cb, 6 ABBA rounds, medians; lm_head 0.997-1.001x, untouched K=2048 path)

- G0: 7 shapes x 40 byte-identical; lmh_dbg histograms zero; gtail (n % 4 != 0 on K = 4096/512/1024/2048,
  grids 0/1/3/16) IDENTICAL. G1: 6/6 byte-identical to 2fd83cb.
- **ABBA decode32 short-ctx x8, b4 (levers 1 + 2)**: tokens identical, no barrier aborts.

| A -> B | short4 tok/s | chat tok/s |
|---|---|---|
| vcgen 39629e7 -> b4 | 187.57 -> **203.62 (1.086x)** | 189.16 -> **203.39 (1.075x)** |
| b1 -> b4 (lever 2 alone) | 190.56 -> 202.61 (1.063x) | 193.42 -> 202.10 (1.045x) |

- Profile b4 (us/token): token 5384 -> 5063; O-proj 898 -> 753; MLP 1305 -> 1140; QKV 743 -> 732;
  lm_head 2226 -> 2236.

Net on dusk short ctx: d1f6208 ~175 -> 2fd83cb ~188 -> 39629e7 ~190 -> b4 ~203 tok/s, all byte-identical to
the 2fd83cb numerics, so the spec (a) matrix is unchanged. wrap512 / ctx512 / R9700 full-Flint rows: albion.

## 3. Lever 3: barrier cuts (b5/b6, parked) and segmented K=2048 gemv (b7, adopted)

**b5/b6, barrier cuts:** every WG computes the deterministic top-K itself (the grid barrier becomes
`__syncthreads`), and the residual add is fused into the O-proj / dense-down gemv store
(`out = out + v`, same float add). That is -16 barriers/token on mid (98 -> 82). G0 + G1 6/6 byte-identical.
- b5 (ACC as a template parameter): 0.943x / 0.943x vs b4. It instantiated two more gemv bodies:
  mid_generate + callees went 11 763 -> 13 107 instructions.
- b6 (ACC as a runtime flag): 0.991x / 0.976x vs b4.

On dusk mid-8L a barrier costs ~1.8 us, so 16 of them (~30 us/token) don't pay for the extra work and
layout change. Parked as `variant_b6_barriercuts_ALBION_CANDIDATE.hip`: on full Flint (49 layers, TOPK 6,
wider grid) barriers cost more. Measure there.

Hot-code observation (mid_generate + gemv callees): vcgen 13 747, b4 11 763 (faster), b5 13 107
(slower). Together with rec-6ob.1's unexplained O-proj slowdown, this points at instruction-cache
sensitivity of the megakernel. Not proven; keep the hot code small (runtime flags over template variants).

**b7, segmented K=2048 gemv** (on b4): matrices that share the input x run as one round-robin quad stream
over a per-WG LDS segment table (`Q4Seg sh_seg[2*TOPK+2]`) with ONE x preload: Q/K/V (3 segments) and all
routed + shared gate/up (2*TOPK+2). There is no new instantiation; with no table, the quad assignment is
exactly the single-matrix one. Per-row arithmetic is unchanged.
- G0 PASS (7 x 40); G1 6/6 byte-identical to 2fd83cb.
- ABBA vs b4: short4 201.94 -> **209.06 (1.035x)**, chat 200.90 -> **207.30 (1.032x)**; tokens identical,
  no aborts.

**Net, all byte-identical to 2fd83cb:** 39629e7 ~188-190 -> b7 ~207-209 tok/s, about **+10-11%** on dusk
short ctx (b4 1.075-1.086x x b7 1.032-1.035x).
