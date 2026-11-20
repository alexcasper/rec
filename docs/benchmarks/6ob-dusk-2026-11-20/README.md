# rec-6ob (rec-bw) step 1: fused q4 gemv load path and megakernel phase profile (dusk 890M, 2026-11-20)

Claude kernels lane. Host: dusk, Radeon 890M gfx1150. All launches were milliseconds (standalone gemv)
or ~0.23 s (megakernel, prompt4 + 36 tokens). No ctx512 or wrap512 launches: on this display-driving
iGPU a 3.77 s mid-8L launch wedged the compositor (rec-u9q incident, 16:03Z). Cotenants: flint :8080
and E4B :8090 up and idle; llama-qwen disabled. busy<5 gate before every round; state on every line.
Native kernels ran without HSA_OVERRIDE_GFX_VERSION. Kernel source: rec/gemini-kernels HEAD
(identical to run-of-record d1f6208 for the mid build, rec-u9q G0).

## 1. ISA (hipcc -O2 --save-temps -Rpass-analysis=kernel-resource-usage; `isa/build.log`)

- `flint_q4_gemv_op<2048|4096|1024>` is NOT inlined. It is an out-of-line function with 42 call
  sites, so every projection pays a call, and the pointers lose their address space:
  - loads are `flat_load_*`, not `global_load_*`;
  - 15 callee-saved VGPRs spill to scratch per call (184 B/lane scratch in every kernel);
  - all kernels are at 248 VGPRs, occupancy 4.
- Inside the K=2048 body each row sits behind `if (row < n)`. The wave issues one row's loads
  (4 x b64 weights + 8 x u16 scale/bias per lane), waits `vmcnt(0)`, then starts the next row: about
  1 KB in flight per wave. Latency-bound.
- Scale/bias are fetched as 2 x u16 per lane per group, so there are twice as many load
  instructions as weight loads. `simd_sum` lowers to `ds_bpermute` (LDS crossbar), not DPP.

## 2. Megakernel phase profile (`mk_prof.py`, MK_PROFILE build = `mk_profile.patch`)

Worker 0 timestamps (wall_clock64, 100 MHz) its workgroup's arrival and the device-wide release at
each of the 140 barriers per token. Median of 3 launches x 36 decode positions, workers=16, mode 0.
- Instrumentation overhead: 0.235 s vs 0.229 s launch (2.6%); tokens identical to production.
- In-kernel decode: 5865 us/token (170.5 tok/s at short context; the 120.78 run of record is after
  512 ctx).
- Barrier wait totals only 309 us (5%). Barriers are not the cost.

| section (all 8 layers) | barriers | us/token | share |
|---|---|---|---|
| lm_head (K=2048 fast path, ~77 GB/s vs ~82 ceiling) | 1 | 2234 | 38% |
| MLP (dense + MoE) | 88 | 1363 | 23% |
| O projection (K=4096 generic path, x global) | 16 | 1314 | 22% |
| Q/K/V (K=2048) | 8 | 743 | 13% |
| attention, norms, KV write, argmax, embed | 33 | 197 | 3% |

Per MoE layer (`prof_w16.json` per_barrier):
- QKV: 93 us (~60 GB/s)
- **O-proj: 152 us (~30 GB/s)**
- expert gate+up: 24-27 us each
- expert down (K=512, x global): 16-19 us
- shared gate+up: 39 us
- shared down (K=1024, x global): 31 us
- elementwise phases: 1-2.5 us each (the barrier floor)

## 3. Lab microbench (`q4_lab.hip` / `q4_lab.py`; `lab_r1.*`, `lab_r2.*`)

- V0 is the production `flint_q4_gemv_host` from a .so built alone from the untouched source.
- Variants run on the same seed-42 inputs (run_q4_bench's). The A9 timing method and bandwidth
  formula are used.
- Every variant's fp32 output is compared byte-for-byte against V0.
- V9 is a pure streaming read of the same weight bytes (the ceiling).

Round 2 (`lab_r2`) uses the megakernel's grid (16 x 256) with x as an fp32 GLOBAL buffer, as in the
megakernel. 4 ABBA rounds, all gated clean, all variants byte-identical:

| shape | op as in megakernel (V10) | V11 batched rows | speedup | stream ceiling |
|---|---|---|---|---|
| o_proj 2048x4096 | 32.97 GB/s | **61.86** | **1.88x** | 77.9 |
| shared_down 2048x1024 | 71.84 | 99.90 | 1.39x | 151 |
| expert_down 2048x512 | 59.08 | 65.70 | 1.11x | 80 |
| expert_gate_up 512x2048 | 76.51 | 82.62 | 1.08x | 82 |
| q_proj 4096x2048 | 60.44 | 63.38 | 1.05x | 80 |

- V11 keeps production's block-outer order and per-row fp32 operation order. On full row-quads it
  issues each block's 4 row loads together, with no per-row guards; the tail keeps the guarded form.
- V10 (33 GB/s) reproduces the in-megakernel O-proj (~30 GB/s), so the lab is a valid proxy.
- V12 (x staged in LDS) adds nothing over V11.
- Round 1 caveats:
  - At default grids, small shapes are launch-overhead-bound: 6-16 us launches, so even the stream
    ceiling reads only 55-85 GB/s.
  - lm_head is already at 73-77 GB/s vs an 81-82 GB/s ceiling; variants gain only 1-9%.
  - **The inlined V1/V3 were NOT byte-identical on o_proj (K=4096)**, while V2/V4/V11 were. The
    likely cause is fp-contract: `s*a + sum*b` can be fused either way, and inlining changes the
    choice. Any inlining change to the megakernel must prove G0; it cannot assume it.

## Next (proposed)

1. Port V11's batched-row structure into `flint_q4_gemv_op`'s generic path (O-proj K=4096, down
   K=512/1024) and V2's into the K=2048 path. Keep the call structure, so contraction context is
   unchanged and only load scheduling moves.
2. G0 on dusk: short4 + chat, byte-identical vs run-of-record, all builds. wrap512 G0 and ctx512
   timing need a headless card (albion is off).
3. Re-profile; ABBA short-ctx timing on dusk.

Projected from the lab ratios: O-proj ~152 -> ~80 us/layer, about 0.6-0.7 ms of 5.87 ms/token
(~11%) on mid-8L. On full Flint the expert-heavy MLP phases weigh more; the R9700 confirm needs albion.

## 4. Step 2: porting batched loads into the run-of-record gemv (G0 FAILED; kernel NOT changed)

Patch: `batched_loads_attempt.patch` (against HEAD; NOT applied). Logs and scripts: `s2/`. dusk solo,
launches <= 0.26 s, cotenants idle, busy-gated. G0 = short4 + chat x {fp32kv, fp32_bf16kv, bf16kv}
with experts, new vs the rebuilt untouched d1f6208 source.

- **Bonus G0 for rec-u9q:** the committed config refactor (HEAD) is byte-identical to d1f6208 on
  gfx1150 (short4 + chat, fp32kv, every array). That dusk G0 had been cut short by the 16:03Z wedge.
- **Attempt 1 failed G0 at 1e-1 relative.** Bisect:
  - generic-path hunk: 6e-7;
  - K=2048 hunk: 0.7.
  Cause of the K=2048 failure: the batched path stored `uint2` into a register array and passed
  `&wq[..]` to `dot16`, which re-reads it as `uint32_t*`. That type-puns private memory: 3-6 of
  150144 lm_head rows came out wrong, always the 4th row of a quad, at any grid (`s2/lmh_dbg.py`).
  The lab variants used the same idiom and happened not to hit it. Fix: `dot16v(u01, u23, ...)` on
  packed words by value, with `dot16` as a wrapper. Afterwards lm_head is byte-identical at every
  grid and `--q4-bench` passes.
- **Attempt 2 (by-value) still fails G0.**
  - fp32-KV differs at 6-7e-7 relative: last-bit fp-contraction differences. hipcc defaults to
    `-ffp-contract=fast`, so restructuring the loads changes which mul+add pairs get fused.
  - The bf16-KV tiers amplify those bits through KV rounding and router flips into token
    differences (rel up to 0.59).
  - The restructure also raised the gemv to 256 VGPRs plus 900 B/lane scratch spills; standalone
    o_proj slowed to 0.552 ms from 0.177 ms.
- Conclusion: under context-dependent contraction, a load-schedule change cannot be proven
  byte-identical in the megakernel; the lab's identity did not transfer. Decision needed (see the
  rec-6ob bead): either a one-time explicit-contraction re-baseline signed off under spec (a), or
  waive G0 for this change in favour of the fp32-KV tier vs the CPU reference.
