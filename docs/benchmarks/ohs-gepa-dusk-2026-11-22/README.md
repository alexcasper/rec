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
