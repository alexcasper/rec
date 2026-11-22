# rec-cnk — full-config depth correctness: onset profile + kernel-vs-reference discrimination (2026-11-22)

## Verdict
**There is no systematic depth error, in the kernel or in the torch reference.** The 7.41e-2 depth-8192 "failure" (default and
split-K alike) is **one fp32 routing tie**: layer 3, position 3232 of the seeded prompt, a k/(k+1) exchange at a selection-score gap of
**4.5–4.8e-7 (2^-21, one fp32 ulp)**. The kernel and torch break the tie differently. Spec (a)'s fp32-KV policy (rec-tan.4, set on mid:
"any declared deviation in the fp32-KV tier is a FAIL") then reports the **unforced** evaluation. On this degenerate repeating
trajectory the unforced reference diverges by 2.5e-2 at 4096 and 7.4e-2 at 8192. Forced to the kernel's own routing, the kernel matches
the reference at ~1e-6 to 3e-5.

## Discriminator design
1. **Routing removed as a variable:** every comparison below forces the reference to the kernel's emitted experts (`step_experts`).
2. **Two reference paths:** R1 = stock single full-sequence forward (the gate's reference); R2 = chunked prefill (1024) through
   KVCache (sliding ring + growing full-attn cache) then one-token teacher-forced decode, i.e. the kernel's launch structure.
   If the reference were wrong at depth, R1 and R2 would split, or one of them would side with the kernel.
3. **Per-position profile from nested prompts:** depth-D prompts are prefixes of the seeded 8192 prompt, and continuation emits are
   bit-identical to single launches (M3.1). So the 8192 emit's `step_experts` IS the per-position routing profile for every D. Emits at
   3072/4096 add logits rows at those depths to confirm.

## Results
**Offline (no window), default build @8192**, `default8192_ref_paths.json` / `default8192_routing.json`:
| forced to kernel routing, 25 rows | step 0 | max |
|---|---|---|
| kernel vs R1 (stock reference) | 1.1e-6 | 1.7e-4 (rows 15, 21; top-2 margins 4.6 / 7.3, argmax equal) |
| kernel vs R2 (chunked reference) | 1.1e-6 | 2.2e-4 (same rows) |
| R1 vs R2 (reference vs itself) | 3.8e-7 | 4.8e-5 |
Routing deviations over 8216 positions × 47 MoE layers (386k decisions): **1** (L3@3232, gap 4.77e-7, k/(k+1)).

**Window 21:27:17–21:41:50Z** (:8080 down; script 21:27:23–21:38:23Z, 11 min, 4 emits rc 0, longest launch 0.194 s; ops_flint in-band
29.83 tok/s after restart, ~11.7 min down; cotenants per line in `window.log`, display active):
| fp32kv, nested seeded prompt | routing deviations (gap) | forced max | strict spec (a) |
|---|---|---|---|
| default @3072 | 0 | 2.6e-5 | **PASS** 4.3e-6 / 2.6e-5 |
| split-K @3072 | 0 | 2.8e-5 | **PASS** 5.0e-6 / 2.8e-5 |
| default @4096 | 1: L3@3232 (4.5e-7) | 2.8e-5 | FAIL 2.48e-2 / 7.95e-2 (unforced) |
| split-K @4096 | 2: L3@3232 (4.5e-7), L26@3255 (4.8e-7) | **4.5e-6** | FAIL 2.48e-2 / 7.95e-2 (unforced) |
| default @8192 (earlier window) | 1: L3@3232 (4.8e-7) | 1.7e-4 | FAIL 7.41e-2 (unforced) |
Onset = 3232: the first fp32-ulp router tie on this prompt. The strict tier holds up to it in both builds. After it, both builds give the
same unforced numbers (2.476e-2 at 4096), because the reference's own trajectory is what changes.

## What remains open (gate design, not kernel)
- **Zero-deviation fp32-KV rule:** with ~47 router decisions per position, an exact fp32 tie (gap of 1 ulp) appears within a few
  thousand positions. At depth the rule fails correct kernels. Candidate amendment (proposal for Alex via rec-kcb, NOT enacted): at
  full config, allow declared k/(k+1) exchanges at gap <= 2e-6 (~4 fp32 ulps; observed 4.5–4.8e-7) and judge against the reference
  forced to the kernel's routing; any wider deviation still fails.
- **Forced 1e-4 at 8192:** 2 of 25 rows reach 1.3–1.7e-4, while the reference's own reorder noise is 4.8e-5 there. Up to 4096 every row
  is under 2.8e-5. Whether the tier needs a depth allowance at 8192 is a separate question for the same ruling.
- **Split-K vs default:** split-K is never worse than the default here (4096 forced 4.5e-6 vs 2.8e-5), and at depth runs ~2.7× faster
  (21.55 vs <=~8 tok/s @8192). It's the natural follow-up candidate under a depth-aware tier once Alex rules.
