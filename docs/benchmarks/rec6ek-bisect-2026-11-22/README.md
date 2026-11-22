# rec-6ek: full-Flint correctness bisect (dusk window 2026-11-22 14:49:53-14:51:30Z, :8080 down ~2.5 min)

Spot: fp32kv short4, prompt 4 + 3 tokens, ONE launch per build (7 positions, longest 0.247 s), 16 workers; bank streamed by
`scripts/bisect/bank_stream_shim.cpp` (LD_PRELOAD; 256 MiB chunks); scored on dawn vs torch fp32 CPU reference (flint-int4,
mmap). Raw npz: dusk ~/rec-ve/bisect/win-20261122T144953Z/ (copied to dawn ~/rec-data/bisect/win/).

| build | lever commit | verdict | step rel | forced rel | tokens | mid_generate_fp32kv instr (total .s) |
|---|---|---|---|---|---|---|
| e3c5ed6 | rec-u9q config-driven FULL (albion GREEN) | PASS | 1.62e-6 | 1.42e-6 | 19629 52387 332 | 8283 (37590) |
| 2fd83cb | rec-6ob explicit-contraction dot16v + bounded-unroll generic gemv | **FAIL** | 2.30e-2 | 1.53e-4 | 19629 52387 **3460** | 6580 (28547) |
| 39629e7 | rec-6ob step 4 LDS-staged generic gemv | PASS | 1.55e-6 | 1.29e-6 | = | 6600 (28990) |
| 5d36dc9 | rec-9t8 L1 batched MoE + L2 pipelined generic gemv | PASS | 1.55e-6 | 1.29e-6 | = | 6581 (27090) |
| 24f167d | rec-9t8 L3 segmented K=2048 (Q4Seg) | PASS | 1.55e-6 | 1.29e-6 | = | 6366 (26547) |
| 2b98737 | GEPA knobs ee26f7a..12bf9d6 + SHRINK promote | PASS | 1.55e-6 | 1.29e-6 | = | 9471 (34776) |
| 5363ad5 (HEAD) | + M3.1 continuation, M3.2 split-K (default off), create2/load_bank | PASS | 1.55e-6 | 1.29e-6 | = | 9518 (34919) |

39629e7 through HEAD are bit-identical to one another (same rel values to 17 digits). Instruction counts don't track the
fault (2fd83cb isn't a size outlier).

Findings:
1. HEAD's default build is CORRECT at full config (1.55e-6, 0 routing deviations): no lineage regression.
2. 2fd83cb was wrong at full shapes and 39629e7 fixed it, consistent with its mid bf16-tier miss on plain/00. That's history only;
   unless it is intermittent (see 4).
3. The failing M3 window spot (6.5e-3 forced, row 0 already 6.4e-4) came from `flint_full.so` = HEAD source + `-DGEPA_ATTN_SPLIT=1`,
   through the driver path (create2/load_bank streaming, 2 continued launches). Driver path + continuation + split all pass on mid
   (1.14e-6). The remaining suspects are full-config-only: the split build at full shapes, the >4 GiB streamed load, or an
   intermittent fault.
4. Discriminator window staged (`scripts/bisect/m3_bisect2_dusk.sh`, ~90 s): reproduce the window config; the same .so via shim x3;
   HEAD via shim x3 (determinism); 2fd83cb x3 (race vs defect); HEAD via the driver path.

## Discriminator window (2026-11-22 14:59:17-15:00:41Z, :8080 down ~2 min, coordinator-approved)

Raw npz: dusk ~/rec-ve/bisect/win2-*/ (dawn ~/rec-data/bisect/win2/). Scored on dawn, fp32-KV tier 1e-4.

| run | build / load path | verdict | step rel | forced rel | row 0 forced | routing devs | repeats identical? |
|---|---|---|---|---|---|---|---|
| reproduce | flint_full.so (HEAD + GEPA_ATTN_SPLIT=1) / driver streamed load | FAIL | 2.61e-2 | 6.53e-3 | 1.05e-3 | 6 | - |
| shim x3 | flint_full.so / shim | FAIL x3 | 2.29-2.43e-2 | 2.5e-3..1.1e-2 | 3e-4..1e-3 | 2-3 | **NO** (all 3 differ; token 3 -> 3460) |
| HEAD x3 | full_k_5363ad5 (default) / shim | PASS x3 | 1.55e-6 | 1.29e-6 | 1.10e-6 | 0 | yes, bit-identical |
| 2fd83cb x3 | full_k_2fd83cb / shim | FAIL x3 | 1.7e-4..1.8e-2 | 1.5e-4..6.9e-4 | 1e-6..2.7e-5 | 0-2 | **NO** |
| HEAD via driver | full_k_5363ad5 / driver streamed load | PASS | 1.37e-6 | 1.17e-6 | 1.17e-6 | 0 | rows 0-2 bit-identical to shim-loaded HEAD |

Reading (for the coordinator ruling):
- The streamed-load driver path is CLEARED: HEAD through it passes and matches the shim-loaded HEAD bit for bit.
- The fault is the -DGEPA_ATTN_SPLIT=1 BUILD at full config: nondeterministic run to run, wrong from row 0. At short4 (context <= 11) the split
  branch is never taken (threshold 64), so the race is not in the split math. The extra code changes register allocation
  (scratch 192 vs 144) and timing, which exposes a latent ordering hazard. 2fd83cb shows the same nondeterministic class, and 39629e7
  may have masked it rather than fixed it. On mid-8L the split build is deterministic and passes, so the hazard is full-shape-only.
- Consequence: every M3 megakernel row (34.40/34.74 @512, 39.08/39.35 short, 21.35 @8192, workers sweep) ran on flint_full.so, a
  build that computes nondeterministically wrong logits. Per the fork, those rows need a re-run on a correct build. HEAD's default
  (split off) is correct for 512/short; an 8192 row needs the race found and fixed (split-K is what makes 8k viable).
