# albion resume runbook (rec-109 / rec-6ob / rec-b3u / rec-xh7)

Fire top to bottom once Alex powers albion on. This is a checklist: each box has an expected gate,
and a failed gate stops that section only. Commits are on `rec/gemini-kernels`.
- **2fd83cb** = numerics baseline (option A re-baseline; differs from d1f6208 by ~1e-6 by design).
- **39629e7** = head (2fd83cb plus LDS-staged generic gemv, byte-identical to 2fd83cb on dusk short4/chat).
- Never use d1f6208 as a G0 reference again.

Sources: a4w `run_a4w.sh`, u9q `alb_g0.sh`/`alb_g12.sh`, b3u `q40_rows.sh`, 6ob §5-§6.

## 0. Bring-up and hygiene (every section depends on this)

- [ ] `ssh albion uptime`, then `rocm-smi --showuse --showmemuse`. Expect:
  - R9700 = card1, HIP device 0 (gfx1201, wave32, 64 co-resident workers).
  - MI50 = card2, HIP device 1 (gfx906, wave64, 60 workers).
  - Display = gfx1036 iGPU.
- [ ] Neither the R9700 nor the MI50 drives a display. So the launches banned on dusk are SAFE here:
  - wrap512;
  - ctx512 decode32;
  - full-config (49L) megakernel launches (3-4 s persistent);
  - long llama-bench depth runs.
  Still verify `cat /sys/class/drm/card1/device/../../boot_vga` != 1 before the first long launch.
- [ ] Cotenants: `pgrep -fa "llama-(server|bench|cli)|flint_full|mid_mk"` must be empty for the card
  under test. Also check p1D's windows on rec-b3u: ask before taking the R9700 if a p1D job is live.
  Log `busy=`, `vram_used=` and `gpu_procs=` per line (the `st()` helper from `q40_rows.sh`).
- [ ] Busy gate before every timed or gated launch: busy < 5 on 3 consecutive 0.3 s samples. A card
  with `power/runtime_status` = suspended counts as idle (the `busy()`/`gate()` helpers in
  `alb_g12.sh`).
- [ ] Env, for every native-kernel shell:
  ```
  export LD_LIBRARY_PATH=/opt/rocm/lib:$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib|head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7|head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2|head -1)/lib
  unset HSA_OVERRIDE_GFX_VERSION        # native kernels NEVER run under the override (rec-tan.5)
  export HIP_VISIBLE_DEVICES=<0|1>      # R9700=0, MI50=1; required, otherwise device 0 is ambiguous
  PY=~/rec-ve/.venv/bin/python
  TC=$(dirname "$(readlink ~/rec-ve/rocm10/lib/llvm/bin/clang++)")
  hb(){ env ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_CLANG_PATH=$TC hipcc "$@"; }
  ```
- [ ] Ship trees: from `/mnt/d/ops/rec-gemini-kernels`, for C in `2fd83cb` and `39629e7`:
  ```
  git archive $C engine | ssh albion "mkdir -p ~/rec-ve/resume/$C && tar -C ~/rec-ve/resume/$C -xf -"
  ```
  Use a tar pipe, not scp braces.
- [ ] State on albion disk. Verify each sha before use:
  - bank `~/rec-ve/u9q/flint-int4.bank`: sha prefix a9fa4b17 (`bank.sha256` in the u9q record);
  - HF ref `~/rec-ve/u9q/flint-int4` = `RecursiveSI/flint-int4@3b93f905`;
  - mid ckpt `~/rec-ve/r109/ckpt/mid-prod`: sha prefix f494e7885d10;
  - GGUFs in `~/rec-ve/b3u` (`gguf.sha256`; f16 cfb9ed29…, plus pure q4_0);
  - llama.cpp HIP build `~/llama.cpp/build` (513ad7ae/2660).
- [ ] If anything breaks the host, stop and route to Uther (inf_albion). Don't debug the host here.

## 1. rec-109 correctness suite, R9700 gfx1201 + MI50 gfx906 (commit: 39629e7)

rec-109 closed at 4d6e6ec/074ef0a (d1f6208 numerics). This re-runs it on the new head, because both
the gemv numerics and the LDS staging changed. Template: `a4w-albion-2026-11-20/run_a4w.sh` with
`SRC=~/rec-ve/resume/39629e7/engine` and `CK=~/rec-ve/r109/ckpt/mid-prod`. Run the arches in
sequence ("gfx1201 0 card1", then "gfx906 1 card2").

- [ ] Build per arch: `hb -shared -fPIC -O2 -Wall --offload-arch=$ARCH flint_mid_megakernel.hip -o flint_mid.so`.
  Gate: 0 errors.
- [ ] Occupancy probe (`D.query_occupancy`). Gate: gfx1201 = 64, gfx906 = 60.
- [ ] Tiny megakernel, 11 configs (`run_tiny.sh` loop). Gate: **11/11 MATCH** per arch; cfg09
  near-tie accepted; max rel about 9.1e-4.
- [ ] Mid suite, workers 16 and WMAX, cases short4,chat,wrap512 (wrap512 is safe here):
  `mid_mk_driver.py --ckpt $CK --emit-suite $OUT/suite_w$W --builds fp32kv,fp32_bf16kv,bf16kv --emit-experts --workers=$W`.
- [ ] Spec (a) matrix (torch fp32 on CPU, f0698a5 pin):
  `CUDA_VISIBLE_DEVICES="" PYTHONPATH=$SRC/src $PY $SRC/validation/megakernel_mid/mid_mk_check.py matrix --ckpt $CK $OUT/suite_w$W`.
  Gates:
  - fp32kv PASS (step_rel <= 1e-4 tier; expect ~1e-6), tokens and expert routing exact;
  - fp32_bf16kv PASS (1e-3 tier);
  - production bf16kv judged on tokens exact plus declared near-ties (expect ~3.9e-3, as on dusk; not a sign-off build).
- [ ] Worker invariance: suite_w16 vs suite_wWMAX npz byte-identical (6/6 per build set).
- [ ] **gfx906-specific risk:** VCgen's LDS staging assumes 32-lane logical warps (8 per 256-thread
  block, slot `sg % 8`). On wave64 two logical warps share a hardware wave and its `wave_barrier`.
  Gate: gfx906 suite byte-identical to gfx1201 for fp32kv, as a4w was. If it differs, re-run with the
  2fd83cb tree. If 2fd83cb matches, the staging is wave64-unsafe: file a bead and pin gfx906 to 2fd83cb.
- [ ] MI50 torch arm, unchanged from rec-109: `~/rec-ve/.venv-mi50-63` (torch 2.9.1+rocm6.3),
  `HIP_VISIBLE_DEVICES=1 python engine/tests/torch_rocm_tiny_check.py --ckpt <cpu-baseline-ckpt> --device cuda`
  (source = rec/claude-correctness). Gate: KV ~1.3e-6, device-vs-CPU ~2.3e-6, bf16 TF <= 1 near-tie flip.
- [ ] Record: `docs/benchmarks/r109-resume-<date>/`. Comment on rec-109 with per-arch tables.

## 2. rec-6ob parked items (bead deferred; reopen when this runs)

### 2a. wrap512 G0 and sign-off, mid-8L (gate commit: 39629e7 vs baseline 2fd83cb)

- [ ] Build both trees for gfx1201 (and gfx906 if section 1 passed). Emit
  `--builds fp32kv,fp32_bf16kv,bf16kv --cases short4,chat,wrap512 --emit-experts --workers=16` for each.
- [ ] G0 = 39629e7 npz byte-identical to **2fd83cb** on all 9 files (compare script in 6ob `s4/run4b.sh`).
  - Expected: 9/9 (dusk showed 6/6 without wrap512).
  - Do NOT compare to d1f6208: about 1e-6 diffs are expected there by design.
- [ ] Spec (a) matrix on the 39629e7 wrap512 files. Gates: fp32kv tokens and routing exact,
  <= ~1e-6; fp32_bf16kv PASS (a4w: wrap512 1.86e-4).
- [ ] Gate failure (non-identical vs 2fd83cb) → stop; report per-row ulp as in `s4/gate7.py`. Do not time.

### 2b. ctx512 / R9700 confirm rows, mid-8L (timed: 39629e7 vs 2fd83cb; 2fd83cb vs d1f6208 for context)

- [ ] `abba_short.py` covers short4/chat only (it caps launches for display GPUs). For ctx512 use the
  driver bench, which is safe on albion:
  `mid_mk_driver.py --bench --ckpt $CK --workers={16,64} --mode 0`, one .so per tree. Order
  A=2fd83cb, B=39629e7, ABBA, at least 4 pairs. Gate busy < 5 before each pair.
- [ ] Rows to fill:
  - decode32@512 and short-ctx differential, R9700 workers 16 and 64;
  - optional MI50.
  Gate: tokens identical A vs B.
- [ ] Expected direction: >= 1.0x. On dusk short-ctx, d1f6208 → 2fd83cb was 1.077x and
  2fd83cb → 39629e7 was 1.01-1.02x. The dusk ratios need not transfer: R9700 has a different
  bandwidth/CU balance.

### 2c. Full-config (49L) effect of the o_proj/expert_down changes (unknown until now)

- [ ] Build `-DFLINT_CFG_FULL=1 --offload-arch=gfx1201` from each tree → `flint_full.so` (as in `alb_g0.sh`).
- [ ] G1: `flint_full_driver.py --bank $BK --ckpt $CK --lib <so> --emit-suite suite_<C>_w16 --builds fp32kv,fp32_bf16kv,bf16kv --workers 16`
  (cases default short4,chat,wrap512).
  Gates:
  - 39629e7 byte-identical to 2fd83cb (9/9);
  - 2fd83cb vs the CPU reference: fp32kv PASS (u9q: <= 2.3e-6, tokens and routing exact);
  - bf16 builds judged per spec (a).
  The CPU matrix takes about 1-2 h with high RSS: run it in the background and check `free -g` first.
- [ ] G2: `flint_full_driver.py ... --bench bench_full_<C>_w{16,32,64}.json --mode 0`.
  Baseline e3c5ed6/u9q: 31.4 / 34.9 / **35.45** tok/s decode32@512.
- [ ] Report: delta vs 35.45 at w64. Recompute the effective GB/s (rec-xh7 put it at ~38 vs
  llama.cpp's ~64). Update the 2.33x gap vs R10 82.78.
- [ ] Record: `docs/benchmarks/6ob-albion-<date>/`. Comment on rec-6ob, then let the coordinator
  close it or re-defer.

## 3. rec-b3u incumbent row: llama.cpp-Vulkan pure q4_0 on R9700 (decision-matrix row)

Status: R9 (UD-Q3 mix, 76.3) and R10 (pure q4_0, 82.78 ± 5.34 / prefill 4517) were measured with
**llama.cpp-HIP** (513ad7ae/2660), not Vulkan (rec-xh7 framing correction). A Vulkan row has not been
measured on albion. No Vulkan build exists there as far as the record shows: **verify first, don't assume.**

- [ ] Check `vulkaninfo --summary | grep -i "deviceName\|driverName"`: RADV on the R9700 must be
  listed. If not, route to Uther (inf_albion).
- [ ] Build: `cmake -S ~/llama.cpp -B ~/llama.cpp/build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build ~/llama.cpp/build-vk -j`.
  - Same source commit as the HIP build (513ad7ae) for a backend-only delta.
  - The host compiler is gcc-wrapper-14.2.1 (clang 22 clashed with glibc on the HIP build; see `b3u/build_llama.sh`).
  - Record the build commit/number.
- [ ] Device pinning: `GGML_VK_VISIBLE_DEVICES=<R9700 index from llama-bench --list-devices>`.
  Verify that the jsonl `devices` field names the R9700, not the MI50 or the iGPU.
- [ ] Rows, per the rec-xh7 protocol (pattern `b3u-q40-albion-2026-11-20/q40_rows.sh`; GGUF
  `~/rec-ve/b3u/flint-int4-Q4_0-pure.gguf`, sha in `gguf.sha256`):
  - decode32@512: `llama-bench -m … -ngl 99 -fa on -p 0 -n 32 -d 512 -r 5 -o jsonl`
  - prefill512: `llama-bench -m … -ngl 99 -fa on -p 512 -n 0 -r 3 -o jsonl`
  - Gate busy < 5 before each; no cotenants; log per line.
- [ ] Optional same-session HIP control: re-run R10 with `~/llama.cpp/build`, so the Vulkan vs HIP
  delta comes from one boot.
- [ ] Post on rec-b3u (p1D's bead). Hand the row to the coordinator/p1E for the matrix. Do not edit
  `run-definitions.md`; another lane owns it.

## 4. Full-Flint 18.5 GB UD-Q3_K_XL vs llama.cpp-Vulkan decision benchmark (rec-xh7 protocol)

Megakernel side commit: **39629e7** if 2c passed (byte-identical, so the same numerics as 2fd83cb),
otherwise 2fd83cb. Incumbent: the as-shipped UD-Q3_K_XL GGUF that p1D used for R9 (path in
`albion:~/models/u9q_q40.sh` / `u9q_inc_mix_*.jsonl`; confirm its sha before running).

- [ ] Rows, all on the R9700 in one boot, ABBA-interleaved between systems where practical:
  - M: megakernel full-Flint, `flint_full_driver.py --bench --mode 0 --workers 64` (+16/32 for the
    scaling curve), decode32@512.
  - V-mix: llama.cpp-Vulkan UD-Q3_K_XL as shipped: decode32@512 (`-p 0 -n 32 -d 512 -r 5`),
    prefill512 (`-p 512 -n 0 -r 3`).
  - V-q40: llama.cpp-Vulkan pure q4_0 (the byte-matched decision row, from section 3).
  - H-mix / H-q40: llama.cpp-HIP controls (R9/R10 re-measured in the same boot).
  - Optional depth 8192 (`-d 8192`) for the incumbent only. This is the 31.1 @ 8k provenance
    question; dusk server measured 26.6-26.8. The flag stays open until measured with one
    methodology.
- [ ] Hygiene: cotenants empty on card1 and card2 (MI50 cotenants share PCIe/host), per-line logging,
  busy gate before every row, no p1D job live.
- [ ] Fidelity caveat carried with each row: the byte-matched q4_0 is 11/16 + 11/16 tokens vs torch
  fp32 CPU (b3u `fid_full_q40.json`). The mix is not byte-matched (~3.5-3.9 bpw vs ~4.5).
- [ ] Verdict line: decision ratio = V-q40 / M at byte parity (prior: 82.78 / 35.45 = 2.33x, HIP).
  State the backend explicitly (Vulkan vs HIP).
- [ ] Record: `docs/benchmarks/xh7-albion-resume-<date>/`. Post the matrix update to rec-xh7 (closed;
  comment only) and rec-b3u.

## 5. Close-out

- [ ] Commit records on `rec/gemini-kernels`. Local only: **no push** without Alex's explicit ask
  (6866868 and later are still unpushed).
- [ ] `cd /mnt/d/ops/rec && bd export -o .beads/issues.jsonl`.
- [ ] Leave albion idle (`pgrep` empty) and ask before powering off.
