# rec-a4w correctness legs: albion R9700 (gfx1201) + MI50 (gfx906), 2026-11-20

Claude correctness lane. I reclaimed rec-a4w after the Gemini seat closed. Runs were
between 15:01Z and 15:12Z on albion, one job at a time, each gated on that card's
`gpu_busy_percent` < 5 for 3 samples at 0.3 s. A runtime-suspended card counts as idle,
because amdgpu returns EBUSY on busy reads while suspended.

These are correctness legs only. The perf-vs-torch-floor numbers come later, under the
benchmark hygiene protocol.

## Setup

| | |
|---|---|
| Tree | `rec/gemini-kernels` `4d6e6ec`: the `d1f6208` run-of-record kernels plus the `f0698a5` cherry-pick (rec-s3m: sign-off reference is torch fp32 on CPU). The tiny-kernel fix below is in the commit that adds this file. |
| Build | System HIP 7.2: `ROCM_PATH=HIP_PATH=/opt/rocm HIP_CLANG_PATH=<nix rocm-toolchain>/bin hipcc`. Tiny: `-shared -fPIC --offload-arch=<arch>`. Mid: `-shared -fPIC -O2 -Wall --offload-arch=<arch>`. No HSA override. |
| Checkpoint | mid-prod, sha `f494e788` (the rec-pp2 production checkpoint) |
| Reference | `mid_mk_check.py matrix`, default reference torch fp32 on CPU, `~/rec-ve/.venv` (torch 2.10.0+rocm7.0) |
| Kernel drivers | torch-free, pinned with `HIP_VISIBLE_DEVICES` |

**Occupancy:**
- gfx1201: 32 CUs × 2 blocks = 64 co-resident workers, wave32;
- gfx906: 60 CUs × 1 block = 60 co-resident workers, wave64.

## Results

### Tiny megakernel, rec-qib 11-config matrix

The matrix is seeds 0–4 × prompts `1,5,6,7` and `1,100,200,300,400,511` at n-out 24 on
the signed bank (10 configs), plus the legacy repro. The checker is
`tiny_mk_np_reference.py`, teacher-forced.

| arch | result | max last-step logit rel | near-ties |
|---|---|---|---|
| gfx1201 R9700 | **11/11 MATCH** | 9.118e-4 (cfg08) | 1 accepted (cfg09) |
| gfx906 MI50, before fix | **0/11**: every launch aborts with `HSA_STATUS_ERROR_EXCEPTION … hardware exception 0x1016`, no output written | — | — |
| gfx906 MI50, after fix | **11/11 MATCH** | 9.118e-4 (cfg08) | 1 accepted (cfg09) |

Every per-config error matches the 2026-11-18 dusk gfx1150 record to the digit.

### Mid-8L megakernel, rec-pp2 sign-off matrix

Builds are `fp32kv` + `fp32_bf16kv` with `--emit-experts`. The reference is
`ref=torch-fp32@cpu` and every run reports `sign-off run: yes`, exit 0.

| arch | workers | result | fp32-KV step_rel (gate 1e-4) | bf16-KV step_rel (gate 1e-3) |
|---|---|---|---|---|
| gfx1201 R9700 | 16 | **6/6 PASS** | ≤ 1.44e-6 | short4 9.24e-4, chat 4.28e-4, wrap512 1.86e-4 |
| gfx1201 R9700 | 64 | **6/6 PASS** | identical to workers=16 | identical to workers=16 |
| gfx906 MI50 | 16 | **6/6 PASS** | identical to gfx1201 | identical to gfx1201 |
| gfx906 MI50 | 60 | **6/6 PASS** | identical to gfx1201 | identical to gfx1201 |

Router near-ties accepted under all-declared, with gaps under 2.5e-4:
- chat L2@pos19, 12→7, gap 4.17e-5;
- wrap512 L1@pos175, 14→18, gap 1.78e-5.

Tokens are exact in every case.

The gfx906 outputs are **byte-identical** to gfx1201's across all 6 npz files. They
really ran on the MI50: emit times were about 1.45× longer, e.g. short4 fp32-KV took
125.5 ms vs 85.8 ms. The kernel reduces in fixed 32-lane logical groups on both wave32
and wave64, it is invariant to worker count, and both builds used the same compiler.
Against dusk's gfx1150 run-of-record outputs (TheRock 10 compiler), the fp32 tier agrees
to ≤ 7.6e-7. See the rec-109 record.

## Fix: tiny megakernel on wave64 (`tiny_megakernel.hip` `warp_sum`)

**Root cause.**
- `warp_sum` reduced with `__shfl_xor_sync(0xffffffffffffffff, v, off)`.
- HIP's `_sync` shuffles call `__hip_check_mask(mask)`
  (`/opt/rocm/include/hip/amd_detail/amd_warp_sync_functions.h`), which traps when the
  mask names lanes that aren't active.
- `rmsnorm` lets only logical warp 0 continue (`if ((threadIdx.x >> 5) != 0) return;`).
  On wave32 RDNA, lanes 32+ are a separate hardware wave. On wave64 gfx906 they are the
  upper half of the *same* wave, so the full-wave mask named exited lanes and every
  launch trapped.
- The mid kernel avoids this because its reductions run with uniform lanes.

**Fix.** `__shfl_xor(v, off, 32)`: an explicit 32-lane segment, no mask check, and the
whole wave on wave32.
- A scratch build with only this change ran 2/2 MATCH on gfx906 before the fix was
  applied in the repo.
- After the fix, gfx1201's 11 tiny outputs are **byte-identical in every array** to the
  pre-fix build. So it is a no-op on wave32; gfx1150 wasn't re-run (see the bead
  comment).

## Files

- `run_a4w.log` / `run_a4w.sh`: the full pass (builds, occupancy, tiny matrix, mid
  suites, sign-off matrices) for both archs. Its gfx906 tiny lines are the pre-fix
  crashes.
- `run_tiny.log` / `run_tiny.sh`: the tiny re-run after the fix, gfx906 then gfx1201.
- `out-<arch>/signoff_w<N>.txt`: sign-off matrices.
- `out-<arch>/emit_w<N>.log`: kernel emit logs with timings.
- `out-<arch>/build_*.log`: build logs.
- `out-<arch>/tiny-fix/tiny_cfg*.log`: per-config tiny logs after the fix.
- `out-gfx906/tiny_cfg01_PREFIX_crash.log`: the pre-fix hardware exception.
- Full npz outputs stay on albion: `~/rec-ve/a4w/out-<arch>/`.
