# rec-3cr full-config soak — dusk, 2026-11-22 (window 16:13:53Z → 16:15:05Z, :8080 back 16:16:27Z)

Coordinator-approved ~2 min :8080-down window, stop/done signalled seat-to-seat to ops_flint (w11:p1Q).
Kit: `scripts/bisect/m3_soak_dusk.sh` (LD_PRELOAD bank-stream shim, 21.02 GiB flint-int4 bank streamed in 8.9 s per build),
`spot_bisect.py --repeat N` (7-position fp32-KV spot, short4 prefix: 4 prompt + 3 decode tokens).
Cotenants per line: :8080 down, E4B :8090 200, qwen3vl :8990/:8991 up (loopback-bound), :8091 inactive, display down;
MemAvailable 16.7–16.9 GiB; longest launch 0.227 s (cap 0.25 s).

| build (all FLINT_CFG_FULL) | sha16 | repeats | distinct outputs | fp32kv spot vs torch fp32 CPU |
|---|---|---|---|---|
| split-K, SHRINK=1 (racer control) | c93fc3a9ad29fe87 | 3 | **3** (tokens 3460/332/3460) | **FAIL** last 3.95e-2 / 1.28e-2 / 5.2e-3, 3–4 routing devs |
| split-K, SHRINK=0 (remedy) | be5d057274f1929d | 8 | 1 | **PASS** last 9.21e-7, step 1.55e-6, 0 routing devs |
| split-K, FORCE_INLINE=1 | b86452cad0e92eff | 5 | 1 (same bytes) | PASS (same) |
| default, SHRINK=0 | a04bcb0f14b62a58 | 3 | 1 (same bytes) | PASS (same) |

All 16 remedy outputs are byte-identical to each other and match the R11r default-build spot (step 1.55e-6).
Scored on dawn with `scripts/bisect/score_soak.py` (one Reference, distinct outputs only).

## Fix landed
`GEPA_SHRINK` defaults to 0 under `FLINT_CFG_FULL` (mid default unchanged at the promoted 1). Instruction streams verified:
mid default identical to before; full split / full default identical to the soaked `full_split_shr0` / `full_def_shr0`.

## Root cause status
First divergence (mid reproducer): O-proj generic K=4096 gemv output, row 0 of row-quads. Excluded experimentally:
barrier visibility, wavefront LDS fence pairs, spill volume. Disasm diff against sunny's class-B patterns (6dbb2bc):
no hit for (a) private escapes, (b) mixed-type VMEM dest sharing, (c) scratch store→reload without vscnt in the quad loop.
Not pinned at ISA level; remedy is codegen pinning per sunny's guidance until sunny-2gg lands.

## Caveat
The spot covers 7 positions; the split branch (GEPA_ATTN_SPLIT_MIN=64) is NOT taken. Here the control races through the
codegen effect, not the split code path. Determinism of the split path at depth on full config still needs a
repeat check (ctx512 / 8192 rows under the driver's determinism gate) in the next window.
