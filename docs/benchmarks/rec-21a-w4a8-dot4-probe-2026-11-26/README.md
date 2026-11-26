# rec-21a step 2: W4A8 dot4 timing probe — cost-model re-check for the conditional GO (2026-11-26)

**Host:** dusk gfx1150 (890M), post-reboot, K=2048 probe (row = 1024 B packed nibbles, 32 g64 groups),
N=4096 rows, one warp per row, lane-per-g64-group. Harness committed at 23ef129 + this dir's `w4a8_probe_run.cpp`
(native C++ runner — python CDLL launches fail on dusk post-reboot with hipErrorInvalidKernelImage; inf_amd owns
that regression, native is healthy). Coordinator env recipe mandatory for every non-interactive launch:
`LD_LIBRARY_PATH=/home/dusk/opt/rocm-10.0.0/lib:<nix libstdc++.so.6 dir>` (rocm-hostlibs lacks libstdc++).

## Schemes
- **P0 baseline:** dequant-FMA on packed nibbles, fp32 activations (today's `flint_q4_gemv_op` inner op).
- **P1 W4A8:** `v_dot4_i32_i8` (signed i8 × signed i8; weights 0..15 are identical under signed interpretation),
  A = per-g16 quantized s8 activations pre-packed as int32 dwords by the quantize pass, B = nibbles unpacked
  in-kernel per dot4 unit (SCALAR lane packing — conservative; vectorized perm tricks only improve it),
  per-(g64,g16) fp32 rescale + activation-sum dot4s.
- **P3 quantize pass:** fp32 K-vector → s8 + per-g16 scales + packed A-dwords, one warp per vector.

## Results (good phase, rc=0 everywhere)

| metric | run A (reps=10, degen s=1/b=0) | run B (reps=50, full affine) |
|---|---|---|
| P0 dequant-FMA | 72.30 µs/gemv | 69.51 µs/gemv |
| **P1 dot4+unpack** | **61.96 µs/gemv (1.167×)** | **61.85 µs/gemv (1.124×)** |
| P1 rel_err vs fp32-A×q ref | 9.7e-8 | 1.3e-7 |
| P3 quantize | 0.16 µs/vector | 0.16 µs/vector (max dequant err 0.0174 = <1 LSB) |

P1 rel_err ≈ 1e-7 = pure fp32 rescale rounding — the integer dot is EXACT (bit-exact vs host per-construction).

## Cost-model re-check (the step-1 conditional-GO condition) — GO CONFIRMED, model REVISED

1. **Scale-partial overhead at g16/g32 is negligible.** Per g64 weight group the dot4 count is constant across
   activation grouping (AG groups × (16/AG product + 16/AG sum) = 32 dot4s); finer grouping adds only
   AG×2 fp32 ops + scale loads (~6 ops/64 elems at g16) — unmeasurable here (degen vs full runs differ only in
   rel_err, not timing). The step-1 worry ("4× the scale partials") does not materialize in the dot4 formulation.
2. **The binding cost is B-side nibble unpack** (~128 scalar ops per g64 group). With the conservative scalar
   unpack P1 already beats P0 by 1.12–1.17×; vectorized unpack has known headroom beyond that.
3. **Quantize pass is free:** 0.16 µs/vector = 0.26% of a 4096-row gemv call, and it amortizes across every
   expert gemv sharing the same activation vector.
4. **Quality requires g16:** step-1 ladder — a8_g16 PASS (−0.07% ppl, 97.64% top-1), a8_g64 FAIL (96.53%).
   The measured grouping overhead being ~zero means the g16 quality win costs nothing measurable.

**Verdict: the rec-21a conditional GO is CONFIRMED with the per-g16 activation recipe.** W4A8-dot4 kernel work is
unblocked; expected gemv-level speedup ≥ 1.12× over the dequant-FMA path (conservative), with the unpack
vectorization and A-sum-to-quantize-pass moves as known follow-on optimizations. Integration caveat: the probe
measures a standalone per-row-warp gemv shape; the megakernel's LDS-pooled multi-row quad path needs its own
ABBA row before any promotion claim.

## Bugs found + fixed en route (kept for the record)

- **A-dword index bug (the 3.918 rel_err):** P1 indexed `qa4[e0>>2]` with the g64-LOCAL element index — every
  lane read activation dwords 0..15 instead of its own group's slice. Fixed to the global index
  `qa4[(g*64 + e0)>>2]`; rel_err dropped 3.918 → 1.3e-7. Coordinator's 20:12Z run hit this same bug.
- **dot4 sign semantics (empirical, corrects the sunny memo's reading):** `v_dot4_i32_iu8` treats BOTH operands
  as unsigned by default (dot4(0xFF.., 0x01..) = 1020, not −4); NEG[0] is NOT required — instead use
  **`v_dot4_i32_i8`** (exists on gfx1150, mods word 0x7c vs 0x1c), whose signed×signed interpretation is
  bit-identical for B ∈ [0,15] and correct for signed activations. Verified with known vectors
  ([1,−2,3,−4]·[5,6,7,8] = −18 exactly).
- **Operator env for non-interactive native launches** (recurring pitfall): plain rocm-env LD_LIBRARY_PATH
  (rocm/lib:rocm-hostlibs) lacks the nix libstdc++.so.6 the HIP image loader needs → hipErrorInvalidKernelImage
  (200) that looks like loader flapping. Always:
  `LD_LIBRARY_PATH=/home/dusk/opt/rocm-10.0.0/lib:<dir of nix libstdc++.so.6>` (recipe: /tmp/kfd-phase-watch.sh).
- This clang (AMD 23.0) has no dot4 builtins — inline asm is the path; v_dot4_i32_iu8 NEG modifiers have no
  llvm-mc operand syntax on gfx1150, which settles the i8-mnemonic choice.

## Step 3/4 megakernel integration status (2026-11-26, commits 96446ca + follow-up)

GEPA_W4A8 knob integrated in the megakernel gate/up segmented gemv (in-place sh_xn quantize:
packed A-dwords + per-g16 scales + per-g16 sums; two-phase LDS barrier; per-(g64,g16) rescale via
v_dot4_i32_i8). Flag-off default build: wrap512 fp32kv md5 fd5d3a234976 = exact pre-patch-source
match (hard rule), det x2.

W4A8 lib (GEPA_W4A8=1, SHRINK=0) after fixes: det x2 (87f7514b62f2), 24/24 token agreement vs the
INT4-fp32-A baseline, last-logit max|d| 0.014 (rel 0.0052) — activation-quantization-level noise.
Launch 0.2486–0.2495 s vs cap 0.25 (W4A8 ≈ +6% per launch vs fp32-A baseline in-megakernel — the
gate/up gemv context reverses the standalone probe's 1.12–1.17×; scalar unpack still the suspect).
Bugs found: quantize pack accumulator not reset between dwords (garbage dwords 1–3 — first-port bug);
A-dword g64-local index (step-3); mixed-scale across two g16 groups under one rescale (first draft).

Remaining for rec-21a step 3/4 gates: ABBA 12/12 vs INT4-fp32-A (dusk window), ppl/top-1 via dawn
scorer with the expert-only g16 fake-quant reference (top-1 ≥ 97%, |ppl| < 1%).

## Superseded / context

- Coordinator's 20:12Z run (P0 137.07 µs, P1 90.77 µs = 1.51×, rel_err 3.918): numerics INVALID (the A-index bug
  above), timing indicative only — it ran under a different load moment (P0 itself 2× this window's).
- The earlier w4a8_capture.sh loop was blind for the same env reason (non-interactive dl_host without the
  libstdc++ dir logged false "loader bad" cycles); killed by the coordinator; inf_amd's kfd-phase-watch
  (/tmp/kfd-phase-watch.log) shows an unbroken PASS streak since the 19:06Z reboot with the correct env.
- dot8_i32_iu4: likely requires i4 activations (fails quality per rec-hyf) — remains an open question for sunny,
  non-blocking.
