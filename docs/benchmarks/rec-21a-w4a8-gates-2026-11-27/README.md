# rec-21a W4A8 step-4 gates — mid-8L, dusk, 2026-11-27 (w17:pQ)

Variant under test: GEPA_W4A8=1 megakernel (routed-expert gate/up inputs quantized per-g16
int8, symmetric absmax/127 RNE; v_dot4_i32_i8; QKV/o-proj/down/lm_head stay fp32-A; KV fp32kv
mode 2 both sides). Baseline: w8base flag-off build (INT4-fp32-A), wrap512 md5 fd5d3a234976
(hard rule: exact pre-patch-source match). Libs: ~/rec-ve/i8/{w8base,w8on}/flint_mid.so at
commits 96446ca→80beee4 source.

## Gate results

| gate | target | result | verdict |
|---|---|---|---|
| determinism ×2 fresh | identical md5 | 87f7514b62f2 / 87f7514b62f2 | PASS |
| hard rule (flag-off default) | fd5d3a234976 unchanged | exact match, det ×2 | PASS |
| ppl \|Δ\| vs torch fp32-A | < 1% | **+0.0126%** (177595.06 vs 177572.62) | PASS (79× margin) |
| top-1 ≥ 97% vs torch fp32-A | ≥ 97% | **96.47%** (n=1528, first flip pos 1) | **FAIL (−0.53 pp)** |
| impl check vs torch g16 fake-quant ref | tight | ppl_rel +0.0040%, max\|dlogp\| 0.182, top-1 within 0.2 pp | PASS (informational) |
| launch cap | ≤ 0.25 s | 0.2486–0.2495 s wrap512; 0.20209 s max in ABBA | PASS (marginal) |

Reference ceiling: torch fp32-A + routed-expert gate/up g16 fake-quant scores **top-1 98.49%,
ppl_rel +0.0087%** vs fp32-A — the recipe itself clears the gate; the kernel loses ~1.8 pp vs
its own reference (96.66% vs R1), i.e. roughly one fifth of the miss is kernel-side deviation,
the rest is the g16 quantization at mid. Kernel-vs-R1 first flip also at position 1 → small
systematic kernel-side deviation exists (candidate: fp32 rounding-order in quantize/dot4 rescale).

## ABBA 12/12 vs INT4-fp32-A (dusk windows, per-row cotenant gate, flint/e4b UP, gated-not-paused)

Window protocol: i8_bench.sh rec-ywy mid protocol (rec-bhz lesson); entries w8base:2, w8on:2,
w8base:2, w8on:2; entries 1–2 skipped (iGPU busy3s≈293, e4b extraction churn), entries 3–4 ran;
r4 ctx8192 hit the 600 s window budget after the 138 s prefill and was re-run standalone (r5).

| row | w8base (INT4-fp32-A) | w8on (W4A8) | delta |
|---|---|---|---|
| short decode | 185.55 tok/s (12/12) | 177.16 tok/s (12/12) | **−4.5%** |
| ctx8192 decode | 33.33 tok/s (12/12) | 16.48 tok/s (10/12, r5; **r6 ABORTED**) | **−50% (noise-qualified)** |
| prefill 8192 | 136.16 s | 138.26 s (r4; r5 measured 162.27 s under churn) | +1.5% |
| max launch | 0.19726 s | 0.20209 s (short); **0.24511 s (r5 ctx) / 0.27027 s (r6 ctx, ABORT)** | cap violation at 8k |

r5 standalone w8on ctx8192 window: 16.48 tok/s, 10/12 stable pairs, max launch 0.24511 s — under
the cap by 2 ms. r6 rerun: **ABORTED — launch 0.270 s > 0.25 s cap at start=8196**. The W4A8
launch cost grows with context depth (wrap512 prefill 0.2486 → 8196 0.270) and the 8k row cannot
complete reliably. Suspected mechanism (flagged, not proven): the WA8 branch inflates the
monolithic megakernel's register pressure, degrading occupancy of the KV-heavy phases in the same
launch — consistent with the cost growing with depth and the short row losing only 4.5%.
e4b :8090 churn was visible during parts of r4/r5 (recorded in row logs); the cap abort in r6 is
not noise-related (clean gate, e4b idle).

## Score provenance

- Scorer: scripts/bisect/score_w4a8.py (torch fp32-A base + router-flag-gated (512,2048) expert
  gate/up-input g16 fake-quant = exact kernel scope; k/v share the shape but precede the router).
- Kernel side: m3_depth_gate --ppl-set eval_set_full_v1.json --ppl-builds fp32kv on the W4A8 lib
  (i8-subset copy; the gepa/h copy is FLINT_CFG_FULL-only). eval_set sha256 1b06d9280fe6e1e9…
- dusk artifacts: ~/rec-ve/i8/w4a8_score.json, ~/rec-ve/i8/w8on_ppl/ppl__fp32kv.npz,
  ~/rec-ve/i8/abba_w4a8_20261127T005311Z/ (summary.log, r3/r4/r5 jsons).
- Cotenant state per row in summary.log (flint:idle/e4b:idle|busy, qwen:down, gpu_busy, MemAvail).

## Verdict (charter)

**W4A8 v1 (g16 gate/up-only, mid) REJECTED for promotion** — three independent grounds:
1. **top-1 gate FAIL**: 96.47% < 97% binding gate (−0.53 pp), despite ppl passing with a 79×
   margin. The g16 recipe ceiling itself is 98.49%; the kernel consumes ~1.8 pp of that headroom.
2. **launch-cap violation at 8k**: 0.270 s > 0.25 s at depth 8196 (r6 ABORT) — the 8k row is
   not reliably runnable; wrap512 marginals (0.2486) do not extrapolate to depth.
3. **negative cost everywhere measurable**: −4.5% short decode, −50%-ish ctx8192 decode
   (10/12, one clean window), +1.5% prefill @8k.

Levers if revisited: g8/per-token gate/up scales (ceiling lift toward 97%+), register-pressure
diet or phase split for the WA8 branch (occupancy/launch-cost at depth), kernel-vs-reference
deviation hunt (first flip at position 1 vs the reference's position 5).

