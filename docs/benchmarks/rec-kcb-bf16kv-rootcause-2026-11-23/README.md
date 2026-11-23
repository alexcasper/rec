# rec-kcb — root cause of the full-config fp32_bf16kv strict-tier divergence (2026-11-23, analysis, dawn only, no window)

**Question.** At full Flint, fp32_bf16kv (pure-fp32 compute, bf16 K/V) fails the rec-pp2 strict 1e-3 tier on wrap512: 2.74e-2 with 148
routing deviations (gate512, split+SHRINK=0 build; the default build reads 4.41e-2). Tie-gap class or real KV-tier drift? Does it change the
bf16-KV quality story, or is it fully covered by the adopted spec (a) (production bf16 judged by tokens + determinism)?

**Semantics check.** The kernel's fp32_bf16kv instantiation is `KVT=bf16, PURE_FP32=true`: no bf16 rounding of q, and K/V are rounded with
`__float2bfloat16` (RNE) after q/k-norm + RoPE. That's exactly where and how the torch reference rounds (`kv_store_dtype`, `.to(bfloat16)`
= RNE, `torch_rocm.py:280-282`). The only kernel-vs-reference difference is fp32 summation order upstream of the rounding.

## Experiment 1: the reference against itself (`scripts/bisect/kcb_sensitivity.py`, `sensitivity_gate512.json`)
Torch reference, teacher-forced on the kernel's wrap512 trajectory (512 prompt + 24), unforced routing. P(eps) multiplies every
k_proj/v_proj output by (1 + eps·s), s = fixed random ±1 per element: a 1–2 fp32-ulp perturbation, the size of a summation-order difference.

| full Flint, wrap512 | logits rel step0 | max | expert-set differences (of 536×49) |
|---|---|---|---|
| fp32 KV: stock vs P(2^-23) | 6.9e-7 | 9.1e-7 | **0** |
| fp32 KV: stock vs P(2^-22) | 5.5e-7 | 1.4e-6 | **0** |
| bf16 KV: stock vs P(2^-23) | **2.14e-2** | 0.197 | **2777**, first at pos 7 |
| bf16 KV: stock vs P(2^-22) | 7.4e-3 | 0.077 | **2826**, first at pos 5 |
| **bf16 KV: kernel vs stock reference, unforced** | 1.07e-2 | 4.7e-2 | 2506, first at pos 5 |
| bf16 KV: kernel vs reference forced to kernel routing | 4.7e-4 | 3.3e-3 | — |
Kernel's 148 forced-metric routing deviations: first at **pos 5 (not 0)**. Layer histogram starts at L3 and grows with depth (compounding,
not a fixed offset). 97% are k/(k+1) exchanges. Gap quantiles (min/25/50/75/90/max) = 2.2e-6 / 6.5e-5 / 1.2e-4 / 2.2e-4 / 3.8e-4 / 1.2e-3.
Those are bf16-scale near-ties, not fp32-ulp ties.

## Experiment 2: ppl noise floor of the bf16-KV reference (`scripts/bisect/kcb_ppl_floor.py`, `ppl_floor.json`, eval_set_full_v1)
| bf16 KV | ppl rel vs stock bf16 reference | max\|Δlogp\| |
|---|---|---|
| reference P(2^-23) seed 1 / seed 2 | +3.90e-3 / +5.77e-3 | 2.21 / 1.51 |
| reference P(2^-22) seed 1 / seed 2 | +8.41e-3 / +5.92e-3 | 2.21 / 2.39 |
| kernel split+SHRINK=0 (gate8k) | +3.38e-3 | 1.84 |
| kernel lop8 (e5d-lop8-full8k) | +6.92e-3 | 2.44 |
fp32-KV control, same perturbations: ppl rel +3.9e-8 / +2.4e-7 / +6.7e-8 / +1.8e-7, max|Δlogp| ≤ 2.1e-4. Without the bf16
rounding the noise is invisible.

Measured against the **fp32-KV reference** (3.28033): bf16-KV reference draws are stock −1.31e-3, perturbed +2.58e-3 / +4.45e-3 /
+7.09e-3 / +4.60e-3 (mean of 5 draws **+3.48e-3**). Kernels: split +2.07e-3, lop8 +5.60e-3 (the coordinator's lop8 0.5565% is the same
comparison).

## Root cause
**bf16 rounding-boundary amplification, not KV-tier drift.**
- fp32 summation-order noise (~1 ulp) moves a few K/V elements across a bf16 rounding boundary. Each crossing is a full bf16 step,
  ~2^-8 ≈ 3.9e-3 relative.
- At full config (49 layers, 128 experts, top-6) that perturbation flips router near-ties from the first few positions, and the flips
  compound through depth.
- The torch reference shows the same behaviour against itself: one ulp of noise yields 2.1e-2 and ~2800 expert-set differences. The kernel
  (1.07e-2, 2506) sits inside that envelope. The fp32-KV control shows the same noise is invisible without the bf16 rounding step (0 differences).
- On mid (8 layers) the amplification has too few layers to compound, which is why fp32_bf16kv passes there (~1.9e-4).
- The "148 devs from pos 0" framing is off. They start at pos 5, and they are a symptom of bf16-scale ties.

## Does it change the bf16-KV quality story?
- **The strict tier: no.** No kernel that isn't bit-exact can meet a 1e-3 logit tier against a reference that misses it against itself by
  ~20×. Spec (a) (tokens + determinism for production bf16) already covers it, and this is the mechanism behind the rec-kcb interim ruling.
  It's consistent with ref_vs_ref (torch fp32-KV vs bf16-KV, 1.85e-2).
- **The ppl story: one refinement.** The kernel's bf16 ppl shift (+3.4e-3 split, +6.9e-3 lop8) lies inside the reference's own perturbation
  band (+3.9e-3 … +8.4e-3), so the kernel adds no measurable quality loss beyond bf16 KV itself. But every perturbed reference lands
  *above* the stock bf16 reference, and the stock bf16 reference sits 1.3e-3 below the fp32-KV reference. The stock reference is a lucky
  draw. The honest cost of bf16 KV at full config is roughly **+0.35% ppl vs fp32 KV (draw spread −0.13…+0.71%)**, not −0.13%.
  Both kernels fall inside that spread (split +0.21%, lop8 +0.56%), consistent with the coordinator's lop8 measurement (0.5565%).
- **Gate-design consequence.** The ppl tier (|Δ| < 1% vs the stock bf16 reference) has only ~1.6e-3 of margin over the reference's own
  worst observed draw (8.4e-3). A correct kernel could fail it by luck. Option for Alex: score bf16 ppl against the fp32-KV reference, or
  against the median of N perturbed references, with the 1% limit. Not enacted; rec-kcb stays parked for the ruling.
