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
