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

## 5. Step 3 (option A): spill fix + explicit contraction, gated under spec (a), then timed (`s3/`)

The kernel change was applied to `flint_mid_megakernel.hip`. dusk solo, every launch <= 0.235 s,
busy-gated, cotenant state on every line: flint :8080 and E4B :8090 up and idle, not paused (standing
protocol); qwen disabled.

**Change (v2)**
- `dot16v(u01, u23, ...)` takes packed words by value, with the fp graph explicit:
  - per 4-nibble group `G = fma(x3,q3, fma(x2,q2, fma(x1,q1, x0*q0)))`;
  - `a = ((G0 + G1) + G2) + G3`;
  - `fma(s, a, sum*b)`;
  - `#pragma clang fp contract(off)` inside.
- `dot16` is a wrapper over `dot16v`.
- Generic path (K = 512/1024/4096): `#pragma unroll 2` instead of a full unroll, and on full
  row-quads each block's four row loads are issued together.
- K=2048 path: original structure.
- Rejected v1 (K=2048 with 2 rows in flight): lm_head failed accuracy NONDETERMINISTICALLY in 3 of 6
  runs (rel ~3e-3), still at 248 VGPR + 92 B scratch, and lm_head was 13% slower (`rep.log`).

**Resources** (`res.sh`; per-function `.set .L*.num_vgpr/private_seg_size`):

| function | HEAD | v2 |
|---|---|---|
| gemv K=4096 | 248 VGPR, 184 B scratch | **113, 0** |
| gemv K=2048 | 248, 64 | 248, 64 |
| gemv K=1024 | 119, 0 | 133, 0 |

The failed attempt patch had 256 / 900 B on K=4096.

**Standalone (`--q4-bench`, 6 interleaved reps, medians, ms)**

| build | expert_gate | q_proj | o_proj | lm_head | FAILs |
|---|---|---|---|---|---|
| HEAD | 0.010 | 0.091 | 0.163 | 2.343 | 0 |
| v2 | 0.010 | 0.090 | **0.123** | 2.325 | 0 |

Target was o_proj <= 0.18 ms; met.

**Determinism and accuracy** (`det.py`): 7 shapes covering every K path, 40 runs each over grids
0 and 16. All runs bitwise identical; max rel vs fp64 9.7e-8 (HEAD: same determinism, up to 1.1e-7).

**Contraction independence** (`gate.log` a): v2 built with `-ffp-contract=off` vs the default gives
bitwise IDENTICAL gemv outputs on K = 512/1024/2048/4096 and lm_head. The gemv arithmetic no longer
depends on fusion context.

**Spec (a) gate** (`matrix_v2.txt`): mid_mk_check matrix, ref=torch-fp32@cpu (f0698a5 pin), v2 suite
short4 + chat x 3 builds, with experts.

| file | result |
|---|---|
| short4_fp32kv | PASS 1.16e-6; tokens exact, 0 routing deviations |
| chat_fp32kv | PASS 1.07e-6; tokens exact, 0 routing deviations |
| short4_fp32_bf16kv | PASS 9.1e-4 |
| chat_fp32_bf16kv | PASS 4.4e-4 (1 declared router tie, gap 4.2e-5) |
| short4/chat bf16kv (production bf16 activations) | tokens exact; 3.9e-3 over the 1e-3 tier |

The production bf16kv level is inherent to that build: HEAD's d1f6208 bf16kv measures 3.7-3.8e-3
against the same reference. Under spec (a) bf16 builds are judged on tokens, which are exact.
wrap512 coverage is absent (deferred to albion).

**Timing** (`abba_short.py`; decode32 short-ctx differential t(P+36) - t(P+4), mode 0, workers 16,
8 ABBA rounds, all pairs valid; production-build tokens identical HEAD vs v2):

| case | HEAD tok/s (pair range) | v2 tok/s (pair range) | speedup |
|---|---|---|---|
| short4 | 175.12 (173.6-177.7) | **188.54** (184.6-191.3) | **1.077x** |
| chat | 174.37 (173.2-175.7) | **187.75** (183.6-190.7) | **1.077x** |

**Re-profile** (`prof_v2_w16.json`):
- token 5865 -> **5391 us** (185.5 tok/s in-kernel);
- O-proj 1314 -> **944 us** (-28%);
- MLP 1363 -> 1282 us;
- lm_head and QKV unchanged;
- barrier wait still about 6%.

**Not covered** (albion off):
- wrap512 G0/sign-off and ctx512 decode32 (the run-of-record metric);
- any new run-of-record row;
- R9700 rows.

The full-config effect of the o_proj/expert_down change stays UNKNOWN until albion returns.

## 6. Step 4: load-schedule lever retried on the contraction-independent 2fd83cb (`s4/`)

dusk solo, standalone gemv launches (ms) and megakernel launches <= 0.22 s. busy < 5 before every
round; flint :8080 and E4B :8090 up and idle, logged per line, not paused (standing protocol); qwen
disabled.

Wider per-lane GLOBAL loads are not byte-identical by construction. A lane's chunks sit 256 B apart;
the warp's b64 loads are already one contiguous 256 B span; widening per lane needs a lane->element
remap, which changes the simd_sum tree. So the wide-load lever was tested as LDS staging instead: the
warp moves 4 rows x 256 B with two b128 loads per lane, then each lane reads back exactly its own 8 B.
Same words, same arithmetic.

Variants on 2fd83cb (`s4/variant_*_REJECTED.hip`; the VCgen change is in the kernel):
- **VA**: K=2048 path, all 4 rows' words/scales loaded before compute.
- **VC2048**: LDS staging on the K=2048 path.
- **VCgen**: LDS staging on the generic path (K = 512/1024/4096).

**Gate 1: byte identity vs 2fd83cb** (`gate7.py`; 7 shapes covering every K path, 40 runs each over
grids 0/16, `run4.log`)

| variant | result | resources (gemv K=2048 / 4096 / 1024 VGPR, scratch) |
|---|---|---|
| VA | **FAIL**, nondeterministic: q_proj 3/40 runs, lm_head 21/40 | 248 + 92 B scratch on K=2048 |
| VC2048 | PASS x280 | 248 + 156 B scratch on K=2048 |
| VCgen | PASS x280 | K=4096 151 / K=1024 164 / K=512 115 VGPR, no scratch |

VA's divergence is NOT fmaf-level. In a fraction of runs it hits 1-2 isolated rows by about 2e-2
relative (1.5-2.7M ulp), and other runs are bitwise exact. That is the same run-to-run corruption
signature as step 3's rejected 2-rows-in-flight K=2048 variant. Both batched K=2048 forms carry
extra scratch spill next to the 64-float x preload. Dropped; not chased further.

**Per-op timing** (grid 16 = megakernel grid; A9 method; 6 ABBA rounds; medians, ms)

| variant | o_proj | expert_down | lm_head |
|---|---|---|---|
| base (2fd83cb) | 0.1109 / 0.1101 | 0.0114 / 0.0115 | 2.2618 / 2.2670 |
| VC2048 | 0.1109 (0.999x) | 0.0113 (1.006x) | 2.3552 (**0.960x**) -> dropped |
| VCgen | **0.1042 (1.057x)** | **0.0104 (1.101x)** | 2.2701 (0.999x) |

(Each variant had its own base series, listed in order.)

**Gate 2 (VCgen, megakernel)** (`run4b.log`):
- short4 + chat x 3 builds with experts: **6/6 npz byte-identical to 2fd83cb**.
- Spec (a) matrix vs torch fp32@cpu (`matrix_vcgen.txt`) is therefore identical to 2fd83cb's:
  - fp32kv PASS (<= 1.16e-6, tokens and routing exact);
  - fp32_bf16kv PASS;
  - production bf16kv tokens exact at the inherent 3.9e-3.
- wrap512 coverage absent (albion).

**ABBA decode32 short-ctx differential, VCgen vs 2fd83cb** (mode 0, 8 rounds, all pairs valid,
tokens identical):

| case | 2fd83cb tok/s (range) | VCgen tok/s (range) | speedup |
|---|---|---|---|
| short4 | 188.28 (185.9-191.0) | 190.15 (189.3-190.7) | 1.010x (inside base spread) |
| chat | 186.21 (180.4-188.8) | 190.53 (188.7-192.9) | 1.023x |

Verdict: VCgen is committed as a small, byte-identical win (o_proj 1.06x, expert_down 1.10x per-op;
+1-2% decode). VA (nondeterministic) and VC2048 (slower lm_head) are rejected. The lm_head (38% of a
token, ~77 of ~82 GB/s) and the K=2048 path have no byte-identical load-schedule headroom left on
dusk. Full-config / R9700 effect still unknown (albion off).

## 7. Step 5 (rec-6ob.1): double-buffered LDS staging on the K=2048 path, rec-6u2 fix direction 3 (`s5/`) — REJECTED

Source: sunny's rec-6u2 ISA analysis (`/mnt/d/ops/sunny/research/rec6u2-gemv-k2048-row-corruption-isa.md` §3).
On top of 39629e7 (VCgen), the K=2048 full-quad path stages each 512-element block (4 rows x 256 B, two
b128 per lane) into one of two LDS buffers. Stage t+1's loads are issued before stage t's compute. The
other buffer is rewritten behind a wave fence pair (`amdgcn_fence(RELEASE, wavefront)` + `wave_barrier`).
The quad's scales and biases (512 contiguous B, one b128 per lane) are staged once per quad, double-buffered
by quad parity. Each lane reads back exactly the words and scales it consumed unstaged, and each row
accumulates j = 0..3 in the same order. Only 8-12 VMEM result VGPRs are live across a stage. Final source:
`s5/variant_vd3_REJECTED.hip`; ISA `s5/vd3_gfx1150.s.gz`. dusk solo; standalone gemv launches are ms long and
megakernel launches <= 0.243 s. busy < 5 before every round; flint :8080 / E4B :8090 logged per line and not
touched (E4B answering again by this run); qwen disabled.

Iterations:
- **vd**: separate `__shared__` buffers. G0 PASS x280. But mid_generate LDS went from 32.9 to 57.5 KiB and
  the standalone gemv kernel to 64 KiB. That made co-residency of the 16 persistent workgroups fragile:
  `sync_all BARRIER ABORT` (arrived 2-4/16) in 3 of the ABBA series (`s5/run5b.log`). The barrier's spin
  cap aborted cleanly: no GPU reset, and busy was back to 3 afterwards.
- **vd2**: one per-warp LDS pool (`q4_lds_pool`, 3 KiB) shared by the generic VC staging and the K=2048 double
  buffer. LDS was back to 32.9 / 40 KiB and there were no aborts.
- **vd3**: vd2 plus a rolled tail loop (n % 4 != 0), because the K=2048 function had doubled to 3286
  instructions. Back to 2207 instructions (vcgen 1736).

**G0, byte identity vs 2fd83cb** (gate7, 7 shapes x 40 runs over grids 0/16) plus lmh_dbg rows%4 / iteration
histograms (lm_head 150144 and q_proj 4096 at 3 grids each; tail shapes 4098 and 150142 for vd3): **PASS on
all three variants**. Zero differing rows in every histogram. Resources K=2048: 248 VGPR. Scratch is 192 B (vd)
or 168 B (vd3), but all of it is callee-saved save/restore at function entry and exit; the loop has zero
scratch ops and zero `s_waitcnt_vscnt`.

**Per-op, standalone (grid 16, 6 ABBA rounds, medians, ms)**

| variant | o_proj | expert_down | lm_head |
|---|---|---|---|
| vd  | 0.1056 (1.058x) | 0.0098 (1.166x) | 2.2264 (**1.024x**) |
| vd2 | 0.1050 (1.054x) | 0.0105 (1.083x) | 2.2244 (1.023x) |
| vd3 | 0.1049 (1.061x) | 0.0104 (1.116x) | 2.2127 (1.023x) |

So VC2048's standalone lm_head loss (0.960x) is recovered (1.023x). o_proj / expert_down are VCgen's
unchanged generic path (s4: 1.057x / 1.101x).

**G1, megakernel** short4 + chat x 3 builds: 6/6 npz byte-identical to 2fd83cb (vd, vd2), so the spec (a) matrix
is unchanged.

**ABBA decode32 short-ctx x8 (mode 0, all pairs valid, tokens identical) — REGRESSION**

| A -> B | short4 tok/s | chat tok/s |
|---|---|---|
| 2fd83cb -> vd  | 186.00 -> 168.64 (0.907x) | (aborted, see above) |
| vcgen -> vd2   | 189.48 -> 169.38 (**0.894x**) | 188.76 -> 169.16 (**0.896x**) |
| 2fd83cb -> vd2 | 185.68 -> 169.69 (0.914x) | 187.78 -> 169.48 (0.903x) |

**Phase profile** (mk_prof, MK_PROFILE build, prompt4 + 36, 16 workers, 3 reps; us/token):

| phase | vcgen | vd2 | vd3 |
|---|---|---|---|
| token | 5384 | 5985 | 5877 |
| 11 lm_head (K=2048, 1173 quads/warp) | 2226 | 2234 | 2238 |
| 9 MLP (gate/up K=2048) | 1305 | 1611 | 1556 |
| 7 O-proj (generic K=4096, code unchanged) | 898 | 1037 | 1003 |
| 3 QKV (K=2048, 1-8 quads/warp) | 743 | 889 | 867 |

Reading:
1. **Megakernel lm_head does not move.** At about 163 MB per 2.23 ms (~73 GB/s vs the ~82 GB/s ceiling) the
   register path already hides latency with 4 waves/SIMD, so double-buffering has nothing to recover in the
   megakernel. The standalone +2.3% does not carry over.
2. **Small-n K=2048 phases regress.** QKV and gate/up run only 1-8 quads per warp. The register path issues
   all 16 row-block loads of a quad together (one latency per quad). The pipeline exposes about one
   latency per stage (prefetch distance 1), plus a prologue per op.
3. **Unexplained:** O-proj (generic K=4096, byte-identical code and instruction count) is also +12-15% inside
   the megakernel but not standalone. Candidates: instruction-cache pressure (rolling the tail cut the
   K=2048 function by 1079 instructions and recovered only about a third of the loss), or L2 / write
   interaction with the preceding K=2048 phase. Not isolated; would need a control variant.

Verdict: REJECTED. It is correct by every gate, but it is a 10% decode regression in the megakernel and gives
0% on its target phase. The kernel stays at 39629e7. Of rec-6u2's fix directions, (3) double-buffering is now
measured as no gain on dusk. (4), register batching with a shrunk preload, was not attempted: lm_head in the
megakernel is already ~89% of the bandwidth ceiling, so its upside is bounded by ~10% of one phase.
