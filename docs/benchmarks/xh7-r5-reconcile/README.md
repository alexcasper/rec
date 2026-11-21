# R5 reconciliation: full-Flint :8080 row + the "31.1 solo @ 8k" verdict

ops_flint support-seat run, 2026-11-21 21:48–21:50Z, dusk (Radeon 890M, gfx1150),
flint :8080 llama.cpp b4-a37cc46 build-vulkan / Vulkan-RADV, server relaunched
2026-11-20 21:47Z via `~/models/flint/serve.sh` (same flags as the step-0 run,
`-c 131072`). Protocol replicated from `xh7-step0/README.md` §2 (commit 2ea0c9f):
`/completion`, random token-id prompt (fixed seed per depth), `cache_prompt` off,
64 greedy tokens, timings from the response. Gate: `gpu_busy_percent` < 5 on
3 samples @ 0.3 s **and** 0 busy slots on :8080/:8090 (`:8091` is down by alex's
directive, so it is not gated and is recorded as down). One addition vs step-0:
a wait-for-idle loop before each run, because an unidentified cotenant produced
GPU bursts (busy=16) that failed the gate twice before any row was accepted;
every recorded row passed the gate cleanly.

## Cotenant state (per coordinator request: decode-focused row while the CPU
server lives)

All five runs ran with flint :8080 + E4B :8090 resident-idle **and** the foreign
CPU-fallback qwen3vl server on :8990 (16 GiB RSS, up since 2026-11-20 20:11Z,
~0% GPU) plus its GPU sibling on :8991 resident. MemAvailable ≈ 6–8 GiB.
Step-0 found idle-resident cotenants make no measurable difference (S0 vs S1
within spread); these rows additionally carry the CPU qwen3vl resident.

## Rows (`runs.jsonl`)

| row | depth | decode tok/s (runs) | prefill tok/s (runs) |
|---|---|---|---|
| full Flint UD-Q3_K_XL via :8080 | 8192 | **26.61** (26.66, 26.56) | 442.1 (441.78, 442.45) |
| full Flint UD-Q3_K_XL via :8080 | 512 | **29.12** (29.25, 29.22, 28.88) | 454.2 (460.59, 449.09, 453.06) |

**Full-Flint :8080 row beside R5:** 26.61 tok/s @ depth 8192 — reproduces the
step-0 R5 row (26.56 S0 / 26.77 S1) within 0.05 tok/s, with the CPU qwen3vl
resident. Prefill 442 vs step-0's ≈ 464 (−5%, within cotenant noise).

## Verdict on the Bedivere-cited "31.1 solo @ 8k"

**Not reproduced at 8k depth. Reproduced-under-X at shallow depth, with a
residual unexplained −6%.**

1. The provenance is now on record: the number comes from ops_flint M1
   (2026-10-21, flint-5yy / `ops_flint/docs/m1-dusk.md`). That doc shows the
   throughput probe ran against a server started with **`-c 8192`** (the
   context *window*) and describes the prefill comparison as **"short-prompt
   prefill"** — the decode number therefore was taken at **shallow prompt
   depth**, not after an 8192-token prefill. The "8k" in the citation is the
   context window, not the depth. The citation conflated the two.
2. Measured here: depth 8192 gives 26.61 (−14% vs 31.1); depth 512 gives
   29.12 (−6% vs 31.1). The step-0 protocol (8k depth) cannot produce 31.1;
   a shallow-depth probe plausibly produced ~29–31 on the October boot.
3. The residual 31.1 → 29.1 gap is attributable to boot-era differences that
   cannot be re-measured retroactively: the Oct 21 boot placed weights in
   shared DDR5 via GTT/host-visible memory (471 MB GTT at M1), while the
   current boot has the 48 GiB BIOS VRAM carve-out (step-0 §1) with weights
   resident in the carve-out, plus intervening driver updates. The magnitude
   (~6%) is inside the documented ±2× swing envelope for this box (rec-ebj).
   The exact M1 request shape, sampler state and sample count were not
   recorded, so a tighter claim is not possible — that recording gap is
   itself part of the finding.

**Recommendation:** retire "31.1 solo @ 8k" as an R5 comparable; use
26.61 @ depth 8192 (this run) or the step-0 26.56/26.77 rows as the
full-Flint llama.cpp-Vulkan incumbent, and cite shallow-depth decode as
~29 tok/s if a shallow number is needed.

Raw: `runs.jsonl`; scripts: `bench_r5_reconcile.sh`, `bench_r5_depth512.sh`
(also in the ops_flint repo `tools/`).
