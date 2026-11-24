# rec-hyf: KV-precision ladder + bf16-KV fix study (full Flint, torch fp32 CPU reference, 2026-11-24)

Harness: `scripts/bisect/kv_ladder.py`. This is the rec-kcb teacher-forced path: torch fp32 compute on CPU, full 49-layer flint-int4,
`eval_set_full_v1` (8 × 192 real-text tokens, sha 1b06d928). Every scheme is a fake-quant at the KV store point (after k_norm and RoPE,
the same point as `kv_store_dtype`), dequantized on read, through an insert-only `Model.kv_quant` hook in `torch_rocm.py` (default
None). All metrics are vs the fp32-KV run, from position 0. Harness noise: a repeated fp32 run gives exactly 0.0. No new data: the
static per-channel scales are calibrated on `eval_set_v1` (566 tokens). Dusk was not used.

## Part A: KV-precision ladder

Bytes per element include fp16 scale overhead. KV MB is full Flint at 8192 context: 14 full-attention layers × 4 kv-heads × 128 × K,V
per token, plus 35 sliding layers × 513-slot rings.

| KV tier | B/elem | KV MB @8192 | max-rel vs fp32 | rel @pos0 | first argmax flip (min / items) | top-1 agree | ppl Δ vs fp32 | expert-set diffs |
|---|---|---|---|---|---|---|---|---|
| **fp32** (run-of-record) | 4 | 543 | 0 | 0 | none | 100% | 0 | 0 |
| **bf16** (RNE) | 2 | 272 | 0.516 | 3.3e-3 | 1 / 8 of 8 | **98.6%** | −0.13% | **7985** |
| int8 per-channel static (calibrated) | 1 | 136 | 0.745 | 6.7e-3 | 1 / 8 | 74.3% | **+104%** | 55442 |
| int8 per-channel, in-sample scales (bound) | 1 | 136 | 0.545 | 7.1e-3 | 1 / 7 | 97.8% | +0.38% | 11653 |
| int8 per-token-head dynamic | 1.016 | 138 | 1.11 | 9.1e-3 | 0 / 8 | 97.7% | +0.63% | 14697 |
| int8 per-token group-32 (q8_0-style) | 1.0625 | 144 | 1.13 | 3.4e-2 | 1 / 7 | 97.3% | +0.26% | 12162 |
| **int8 KIVI** (K per-channel / 128-tok group, V per-token) | 1.016 | 138 | 0.732 | 2.8e-2 | 1 / 7 | **97.7%** | +0.10% | 11228 |
| int4 per-channel static | 0.5 | 68 | 1.13 | 5.6e-2 | 0 / 8 | 72.2% | +107% | 61084 |
| int4 per-channel in-sample | 0.5 | 68 | 0.743 | 8.6e-2 | 0 / 8 | 91.0% | +1.13% | 45934 |
| int4 per-token-head | 0.516 | 70 | 0.770 | 5.7e-2 | 0 / 8 | 85.5% | +2.45% | 58382 |
| int4 group-32 (q4_0-style) | 0.5625 | 76 | 0.753 | 2.9e-2 | 0 / 8 | 92.0% | −1.59% | 45336 |
| **int4 KIVI** | 0.516 | 70 | 0.748 | 5.1e-2 | 0 / 8 | **92.6%** | −1.77% | 42010 |

Answers:
- **int8 does not dominate bf16.** It halves memory again vs bf16 (136–144 MB vs 272 MB at 8192), but every int8 scheme is less
  faithful on every discriminating metric. Best int8 (KIVI) vs bf16: top-1 97.7% vs 98.6%, expert-set diffs 11.2k vs 8.0k, pos-0
  divergence 2.8e-2 vs 3.3e-3. The reason: bf16 error is relative (2^-9 per element at any magnitude), while int8 error is absolute
  (≤ group-max/254). Post-norm K/V channels are heavy-tailed, so most elements are small relative to their group max and int8 rounds
  them worse than bf16.
- **ppl can't rank bf16 against int8.** All int8 non-static schemes land at +0.10..+0.63%. That's inside the bf16 reference's own
  noise band: perturbed bf16 refs span −0.13..+0.71% (eb06650), and bf16-SR seeds span −0.03..+0.36%. int8 does pass a 1% ppl gate.
- **Static per-channel scales are a trap.** Scales calibrated on short prompts clip 2.5% of K and 0.9% of V values, giving +104% ppl.
  Real per-channel int8/int4 needs dynamic group scales (KIVI) or calibration with headroom.
- **int4 floor:** every int4 scheme has |ppl Δ| ≥ 1.1% and top-1 ≤ 92.6% (argmax flips at position 0). int4 fails a 1% gate at this
  model size, and the ppl sign isn't meaningful at that noise level. The best int4 (KIVI / g32) is ~4× more routing churn than bf16.
- (A side note) the bf16 row reproduces the rec-kcb baseline exactly: bf16 stock ppl −0.131% vs fp32 (eb06650 −1.31e-3).

## Part B: fixing bf16-KV

**Disease reproduced first** (eb06650 P(eps): k/v proj × (1 ± 1 fp32 ulp), same seeds). The numbers match eb06650 to 4 significant
figures:

| | fp32 KV control | bf16 KV |
|---|---|---|
| max-rel, stock vs perturbed | 4.6e-5 / 6.5e-5 | 0.392 / 0.369 |
| ppl Δ | +3.90e-8 / +2.38e-7 | **+3.895e-3 / +5.766e-3** (eb06650: +3.895e-3 / +5.766e-3) |
| expert-set diffs | 0 / 0 | 4687 / 4882 |

| fix | vs fp32: ppl Δ / top-1 / max-rel / expert diffs | 1-ulp amplification: max-rel / ppl Δ / expert diffs | verdict |
|---|---|---|---|
| bf16 RNE (as-is) | −0.13% / 98.63% / 0.516 / 7985 | 0.392 / +3.9e-3 / 4687 | disease |
| (1) stochastic rounding, seeded s1 | −0.03% / 97.79% / 0.555 / 9310 | 0.239 / −3.5e-3 / 5005 | **no**: more routing churn |
| (1) SR seed 2 / unseeded | +0.36% / 98.11% / 0.560 / 8912; −0.06% / 98.57% / 0.368 / 8443 | — | seed alone moves ppl 0.4 pt |
| (3) per-channel scaled store | −0.05% / 98.18% / 0.703 / 7689 | 0.413 / +2.7e-3 / 5048 | **no**: bf16 rel. error is scale-invariant |
| (2) router decoupling, ORACLE | −0.05% / **99.93%** / **0.072** / **0** | **0.061** / **+2.3e-4** / **0** | kills flips, but not fp32-comparable, and needs fp32 state |
| int8 KIVI / g32 (for reference) | see ladder | 0.356 / +2.5e-3 / 9083; 0.468 / +1.7e-3 / 8224 | same disease |

- **(1) Stochastic rounding** removes bias but not amplification. A 1-ulp shift crosses a random threshold about as often as it
  crosses an RNE midpoint. It also makes the output seed-dependent: deterministic per seed, but the seed moves ppl by ~0.4 points.
- **(3) Scaled store** is predicted to be a no-op by construction: floating-point relative rounding doesn't depend on scale. Measured:
  amplification unchanged.
- **(2) Router decoupling** is the only lever that touches the mechanism. With bf16 KV and fp32-KV routing, divergence vs fp32 drops 7×
  (0.516 → 0.072), top-1 reaches 99.93%, and the first argmax flip moves from position 1 to 149. Two problems remain:
  - **Its amplification isn't fp32-comparable:** 1-ulp max-rel 6.1e-2 vs 4.6e-5 for fp32. That's 1300× worse; the attention path
    alone still amplifies.
  - **It isn't implementable cheaply.** Flint's router is `topk(sigmoid(W·x) + bias)` on the residual stream (`torch_rocm.py:323`); it
    never reads KV. Its input is downstream of bf16 attention, so fp32 routing decisions need fp32-KV attention state, i.e. fp32 KV
    storage, which defeats the memory saving. The oracle bounds the benefit; it isn't a deployable fix.
- **Why nothing works:** any reduced-precision store is discontinuous. A perturbation ε crosses a quantization boundary with
  probability ~ε/Δ per element, and each crossing injects an error of size Δ. So amplification scales with Δ/ulp for every format that
  stores fewer bits than fp32, whatever the rounding rule or scale. Only bit-exact reproduction of the reference's pre-rounding values
  avoids it, which a non-bit-exact kernel can't do. This independently confirms the rec-kcb ruling (bf16 rejected) and extends it:
  int8/int4 fail the same way.

**Rebench: not run.** The directive's precondition (a fix that drops amplification to fp32-comparable) isn't met by any candidate, so
there's no fixed-bf16 kernel to bench on dusk. fp32-KV stays the run-of-record.

## Recommendation (for Alex)
If a memory-saving KV tier is wanted for long context, gate it on quality vs fp32-KV, not on logit-matching (which is unattainable for
any reduced format). For example: teacher-forced ppl |Δ| < 1% plus top-1 agreement ≥ 97% on `eval_set_full_v1`. Under that gate:
- **bf16** is the best 2× tier: 98.6% top-1, fewest routing flips.
- **int8-KIVI** is a viable 4× tier: 97.7%, +0.10% ppl, ~1 pt more top-1 loss than bf16.
- **int4** is not viable at 1%.

KV memory at 8192 is only 0.54 GB fp32 vs the 18.5 GB weight bank, so at this context length the memory case for any reduced tier is
weak. It strengthens only toward ~64k+ context, where fp32 KV reaches ~4 GB.

## Addendum (2026-11-24): fp8 rows for RDNA4, and the int8 decision

The user decided on int8 KV: it runs on both RDNA3.5 gfx1150 and RDNA4 gfx1201, since `v_dot4_i32_iu8` exists on both. They asked to
also try fp8 for RDNA4. ISA probe (`scripts/bisect/fp8probe.hip`): gfx1201 has `v_cvt_pk_fp8_f32`, `v_cvt_f32_fp8` and
`v_dot4_f32_fp8_fp8`; gfx1150 has none of them (they need fp8-conversion-insts and dot11-insts).

| 1 B/elem tier | ppl Δ vs fp32 | top-1 | expert-set diffs | 1-ulp amplification max-rel / expert diffs |
|---|---|---|---|---|
| int8 per-token-head (kernel v1 scheme, rec-ywy) | +0.63% | **97.7%** | **14697** | 0.468 / 10191 |
| int8 KIVI | +0.10% | 97.7% | 11228 | 0.356 / 9083 |
| fp8 e4m3 (OCP, unscaled, saturating) | +0.19% | 96.0% | 22513 | 0.406 / 8824 |
| fp8 e4m3, per-token-head scale | +2.06% | 97.1% | 20325 | 0.444 / 14539 |
| fp8 e5m2, unscaled | +3.19% | 93.8% | 32714 | — |
| fp8 e5m2, per-token-head scale | +1.47% | 94.9% | 29502 | — |

**fp8 is dominated by int8 at identical bytes.** e4m3's 3-bit mantissa rounds with up to 6.25% relative error; scaled int8 has 127
levels per row. The best fp8 (e4m3) is −1.7 pt top-1 with 1.5× the routing churn of int8, and e5m2 fails a 1% gate. fp8's only
advantage would be RDNA4's fp8 dot4 compute, but depth attention is bandwidth-bound and the bytes are the same as int8. So an fp8
kernel variant isn't worth building; rec-18h is deferred on this evidence.
