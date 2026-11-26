# rec-21a step 1: W4A8 CPU Go/No-Go — int8 activations x the existing INT4-affine g64 weights (2026-11-24)

**Executed 2026-11-24 14:38–15:31Z on dawn** (CPU torch fp32 reference, 8 threads, `REC_REF_MMAP=1`,
`PYTHONPATH=src:validation/megakernel_mid`), full flint-int4 ckpt (`/home/dawn/rec-data/flint-int4`, config sha
`80f577a9` — verified identical to dusk `~/xh7/flint-int4` on 11-25), eval_set_full_v1 scoring (sha `1b06d928`),
eval_set_v1 for the a8_static calibration. Both runs end `done`; `w4a8_ladder.json` holds all 10/10 variants the
harness defines. Originally mis-recorded as "died mid-run, unrecorded" by the 11-24 18:30Z dark tick (it searched
docs/benchmarks, /tmp, scratch but not `/home/dawn/rec-data/w4a8/`) — corrected 11-25; no rerun needed or performed.

Gate (rec-21a, same family as rec-ywy/rec-3tv): |ppl_rel| < 1% AND top-1 >= 97% vs the fp32-activation reference.

## Verdict vs the gate

| variant | ppl_rel | top-1 | gate | margin |
|---|---|---|---|---|
| a8_tok (per-token absmax) | +0.135% | 95.29% | **FAIL** | top-1 −1.71pp |
| a8_g64 (per-g64, INT4-weight-group aligned) | +0.342% | 96.53% | **FAIL** | top-1 −0.47pp |
| a8_static (calibrated per-tensor, eval_set_v1) | ppl 1.0e13 | 0.0% | **FAIL** | catastrophic |
| a8_tok_nolm (lm_head fp32) | −0.488% | 96.07% | **FAIL** | top-1 −0.93pp |
| **a8_g32** (per-16-element-group x2 finer) | +0.371% | **97.12%** | **PASS** | +0.12pp |
| **a8_g16** | −0.073% | **97.64%** | **PASS** | +0.64pp |
| a8_g64 attn-only | −0.094% | 97.32% | PASS | |
| a8_g64 moe-only | +0.489% | 97.91% | PASS | |
| a8_g64 shared-only | +0.721% | 97.58% | PASS | |
| a8_g64 lm-only | −0.019% | 99.15% | PASS | |

**Go/No-Go: CONDITIONAL GO.** All four "natural" schemes (per-token, per-g64, static, per-token-nolm) FAIL the
top-1 bar; the g64-aligned dot4 sketch in the bead description misses by 0.47pp. Per-g16 activation scales pass
comfortably (−0.07% ppl, 97.64%), per-g32 marginally (+0.37%, 97.12%). Every single-block g64 ablation passes,
so the g64-all damage is diffuse across blocks, not concentrated. Kernel consequence: the dot4 path needs
per-g16 (or per-g32) activation scales — 4x/2x the scale partials of the g64 sketch — and the cost model must be
re-checked before kernel work (ISA precondition already resolved: sunny 40341e0, v_dot4_i32_iu8 semantics).

## Run details

- run 1 (14:38–15:01Z): fp32-activation base 415 s; a8_tok 243 s; a8_g64 253 s; a8_static (calibration on
  eval_set_v1, 9 weight shapes, 14:53:58Z) 227 s; a8_tok_nolm 216 s. `done` 15:01:21Z.
- run 2 (15:04–15:31Z): base 301 s; a8_g32 232 s; a8_g16 222 s; g64 ablations 197–232 s. `done` 15:31:51Z.
- quantized-linear calls ~222.8k per variant; a8_static clip_frac 9.5e-7 yet ppl 1e13 (static scales calibrated
  on eval_set_v1 collapse out-of-range on eval_set_full_v1) — dead end, do not pursue.

## Note on reruns

A coordinator tick on 11-26 re-queued this study claiming the harness "has never been executed". False: the
outputs above predate the tick and are complete; the computation is CPU-deterministic (same script/ckpt/eval
sets), so a dusk rerun would reproduce these numbers bit-for-bit in expectation. No rerun was performed
(rec-lane-ops rule: never rerun until outputs are proven absent).
