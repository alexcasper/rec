# rec-ohs GEPA hill-climb, dusk mid-8L (24f167d lineage): first run

Driver: `scripts/gepa_kernel_search.py` (CPU host) + `scripts/gepa_remote.py` (dusk). Knobs: compile-time
`GEPA_*` flags in `flint_mid_megakernel.hip` (ee26f7a); the default build is byte-identical to 24f167d (6/6
suite npz + qgate ppl logp). Pool: `docs/benchmarks/rec-perf-candidates.json` on rec/qwen-aip-prep
(evaluations[], rejected[], pareto_front, search{}); rsynced to dusk `~/rec-ve/gepa/pool/` after every update.

Gates enforced (pool `gates.hard_constraints`): rec-pp2 fp32kv @1e-4 + fp32_bf16kv @1e-3 on short4 + chat vs
torch fp32 CPU; ppl |delta| < 1 % vs incumbent bf16kv on the frozen rec-1f4 held-out ppl items (eval_set_v1
sha 72209796...f464). A verdict is inherited when every suite/ppl npz is byte-identical to the incumbent's.

## iter0 (harness calibration, no mutation): incumbent 24f167d vs a copy of itself, 8 ABBA rounds

short4 **207.73** tok/s, chat **207.18** tok/s (target ~207-209: reproduced); self-speedup 1.0011 / 0.9919, so
noise 0.8 % and acceptance threshold min_gain = 2 x noise = 1.6 % on BOTH cases. Max launch 0.199 s.
Gates PASS: fp32kv 1.16e-6 / 1.07e-6; fp32_bf16kv 9.1e-4 / 4.4e-4; bf16kv production 3.9e-3 (recorded, not
tier-gated); ppl fp32kv 114115, bf16kv 114217.

## Run 1 (max 12 evals, patience 4): stopped by the patience rule after 4 evaluations

| id | mutation (axis) | short4 / chat speedup vs iter0 | hot instr | scratch B | gates |
|---|---|---|---|---|---|
| c001 | GEN_UNROLL 2 (unroll_factors) | 0.976 / 0.979 | 12 640 | 88 | pass (inherited) |
| c002 | GEN_UNROLL 4 (unroll_factors) | 0.873 / 0.876 | 13 520 | 412 | pass (inherited) |
| c003 | K512_PIPE 1 (lds_staging) | **1.015 / 1.009** (below 1.6 %) | 12 012 | 16 | pass (inherited) |
| c004 | GEN_PREFETCH_X 0 (lds_staging) | 1.006 / 1.000 | 11 753 | 16 | pass (inherited) |

Not reached before the stop rule: workers 8 (tile_shapes), TOPK_PER_WG, RESID_FUSE (barrier_structure).
The unroll axis repeats the scratch / hot-code signal (more unroll -> more spill + code -> slower).

## Run 2 (kimi-driven, --proposer llm, max 12, patience 4): stopped by the patience rule after 4 evaluations

Acceptance = pool hard_constraints (coordinator ruling); qgate T1-T4 = pre-promotion evidence only.
Cotenants: flint and E4B were not touched (per the standing order; the only cotenant we own, llama-qwen,
stays disabled). Every round was in the state `flint:idle,e4b:idle,qwen:inactive`; 0 rounds dropped by the
new cotenant-stability rule (8b62e5a).

| id | proposer | mutation | short4 / chat vs iter0 | hot instr | gates |
|---|---|---|---|---|---|
| c005 | kimi (1 x 429, retry ok) | K512_PIPE 1 + GEN_PREFETCH_X 0 | 1.011 / **1.018** (short4 below 1.6 %) | 11 774 | pass |
| c006 | none (429 x2, no glm key) -> grid | workers 8 | 0.932 / 0.936 | 11 911 | pass |
| c007 | kimi (429, retry ok; reply had no JSON) -> grid | TOPK_PER_WG 1 | 0.998 / 0.991 | 11 823 | pass |
| c008 | none (429 x2) -> grid | RESID_FUSE 1 | 1.002 / 0.993 | 11 762 | pass |

kimi 429 total 6 over 8 calls. The first attempt of every call was throttled, and the call rate was about
1 per 3 min, so the trial quota is tighter than the 30 s window. All 6 knobs plus the best pair have now
been tried with none of them above the 1.6 % threshold on both cases. The byte-identical knob neighbourhood
of 24f167d is exhausted at ~207-211 tok/s. Front: 24f167d unchanged.

## Step 2 (rec-1f4, Track B relaxed): g64 group-to-lane K=2048 modes, fed as --candidate lists (9c0d30e, 3e99533)

Modes per call site (GEPA_S2_{MOE,QKV,LMH,DENSE}): 1 = whole g64 group per lane (2x16 B + 1 s/b pair per
row-lane vs 24f167d's 4x8 B + 4 s/b pairs); 2 = half-groups, contiguous fully coalesced 16 B loads;
3 = packed row-pair staged through q4_gen_pool (2 KiB/warp), one whole group per lane consumed from LDS.
Defaults stay byte-identical (8/8 npz). Bench = ABBA x8 vs iter0, 0 rounds dropped (single cotenant state).

| id | candidate | short4 / chat | hot instr | fp32kv max rel | fp32_bf16kv max rel | ppl delta |
|---|---|---|---|---|---|---|
| c009 | MoE grp | 0.967 / 0.967 | 13 556 | 1.09e-6 | 8.9e-4 | +6.5e-4 |
| c010 | MoE half | 0.949 / 0.957 | 13 579 | 1.10e-6 | 9.2e-4 | -1.6e-4 |
| c011 | MoE LDS-packed | 0.959 / 0.967 | 14 775 | 1.09e-6 | 8.9e-4 | +6.5e-4 |
| c012 | lm_head grp | 0.999 / 1.007 | 13 386 | 1.11e-6 | 9.1e-4 | 0 |
| (stop rule: 4 without improvement) | | | | | | |
| c013 | all sites grp (I-cache control) | 1.004 / 1.003 | **11 582** | 1.13e-6 | 9.3e-4 | +3.6e-4 |
| c014 | all sites half (I-cache control) | 0.982 / 0.979 | 11 641 | 1.10e-6 | 9.0e-4 | -1.0e-3 |

All pass the hard gates. None is accepted, so qgate T1-T4 was not run (it is pre-promotion evidence only).
c013/c014 were a separately labelled 2-eval hypothesis test, not a reopened search. Per-site builds keep the
strided GM=0 path live, so each per-site candidate carried +1.5-2.9k hot instructions. With every site
converted (hot code BELOW default), the group mapping is neutral (1.004/1.003).

Reading: K=2048 load-instruction count / coalescing is not what bounds mid-8L decode on dusk. lm_head is at
the bandwidth roof (~78/82 GB/s, so neutral is expected). The MoE/QKV phases are short (~tens of us per
layer-phase) and look bound by per-phase ramp + barrier latency, not issue rate. Staging through LDS adds a
hop and fences on top (rec-6ob.1 prior, confirmed: c011 is worse than c009 at more code).
Margin note: fp32_bf16kv max rel sits at 8.9-9.3e-4 for every build, baseline included (tier 1e-3). Any future
relaxed lever has <= 1e-4 headroom on that tier.

## Latency-side levers (GO 2026-11-22): direction 1 (3edf6bf) and direction 2 (5b9a8cd). Both stopped by the stop rule.

All 8 candidates produce suite + ppl npz byte-identical to 24f167d. Gates were inherited, and the fp32_bf16kv
error is unchanged (no headroom consumed). 0 ABBA rounds dropped; one cotenant state throughout.

Constraint found: sync_all opens with an agent-scope __threadfence (vmcnt(0)), so no load can be held across a
grid barrier. Direction 1 therefore has warps that are IDLE in a light phase warm the L2 (2 MB on gfx1150) with a
contiguous prefix of the next weight phase: attention -> O-proj; router + top-K -> shared gate/up (shared segments
moved first in the quad stream); act -> down; combine -> next-layer Q / lm_head. GEPA_PF = bytes per idle warp.

| id | candidate | short4 / chat | hot instr |
|---|---|---|---|
| c015 | d1 PF 4 KiB | 1.007 / 1.006 | 12 155 |
| c016 | d1 PF 8 KiB | 0.991 / 0.989 | 12 155 |
| c017 | d1 PF 2 KiB | 1.004 / 1.002 | 12 155 |
| c018 | d1 PF 16 KiB | 0.961 / 0.969 | 12 155 |
| c019 | d2 act fold (MoE act in LDS per WG; -7 grid barriers) | 0.989 / 1.002 | 11 825 |
| c020 | d2 act fold + PF 4 KiB | 0.992 / 1.001 | 12 015 |
| c021 | d2 act fold + residual fusion (-15 barriers) | 0.992 / 1.000 | 11 684 |
| c022 | d2 act fold + residual fusion + PF 4 KiB | 0.990 / 1.000 | 11 885 |

Direction 2 as literally specified (gate/up -> act -> down per expert in ONE phase) needs either one workgroup
per expert (3-4 of 16 WGs busy) or split-K down with float atomics, because down needs the whole act vector.
The arithmetic-identical form tested here removes the act grid barrier (and, with RESID_FUSE, the O-proj residual
barrier).

Reading: removing up to 15 grid barriers per token, or overlapping light phases with L2 warm-up, moves decode by
less than the 0.8 % noise. Bigger prefetch windows hurt, because idle warps then finish after the light phase and
delay the barrier. On dusk mid-8L, decode time sits inside the weight phases; barrier/restart latency is a
second-order term.

## Profile-directed candidates 1 + 2 (GO 2026-11-22; profile in ../ohs-profile-dusk-2026-11-22)

Driver change: gates run BEFORE the bench, and every record carries the fp32_bf16kv max error vs the incumbent
(flagged if up). 0 ABBA rounds dropped; cotenants idle/idle/qwen inactive throughout. flint and E4B were NOT
paused (owned by ops_flint; the standing rule is "untouched").

| id | candidate | short4 / chat tok/s | vs 24f167d | fp32kv max | fp32_bf16kv max (margin) | ppl delta | verdict |
|---|---|---|---|---|---|---|---|
| c023 | c1 O-proj split-K, runtime row stride | 189.8 / 188.3 | 0.917 / 0.907 | pass | 9.36e-4 (6.4e-5) FLAG | +8.1e-4 | reject: runtime stride spilled the shared K=2048 loop (scratch 16 -> 128) |
| c024 | c1b same, compile-time stride (O-proj-only instantiation) | 198.0 / 199.3 | 0.957 / 0.963 | pass | 9.36e-4 (6.4e-5) FLAG | +8.1e-4 | reject: slower |
| c025 | c2 MoE down as one virtual K=2048 gemv, weights folded | 207.4 / 206.7 | 0.996 / 0.997 | pass | 9.31e-4 (6.9e-5) FLAG | -0.9e-4 | reject: neutral |

The profile's per-phase "loss vs roof" did not convert into time when the phase was moved onto the K=2048 path.
- O-proj: 2048 rows split into two halves leaves each warp only 2x4 quads, plus an x reload between halves. That is
  less depth than the generic K=4096 stream gives, plus +614 hot instructions.
- MoE down: one stream instead of three calls is neutral. The phase is short (52 us/layer) and appears bound by the
  same per-warp depth (4 quads/warp at N=2048), not by the call structure. vgpr sits at the 256 cap.
- Every relaxed candidate moves fp32_bf16kv up by ~2e-5, leaving 6-7e-5 under 1e-3.
Contingent items (router over all warps, attention split) were NOT started: neither candidate landed.
