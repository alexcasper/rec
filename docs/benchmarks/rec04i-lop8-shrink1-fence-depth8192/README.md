# rec-04i/ruling — full lop8 SHRINK=1+VMEM-fence @8192 bench row (2026-11-24)

**Alex directive via Bedivere: benchmark the unmeasured variant.** Variant = HEAD default build
`-DFLINT_CFG_FULL=1 -DGEPA_ATTN_SPLIT=1 -DGEPA_ATTN_LOP=8` with no SHRINK define (SHRINK=1 + the rec-11b
`gepa_vmem_fence()` at both `q4_rows_tail` call sites), gemini-kernels ee04ff3 source, lib
`ec34a71153bae74b` (VGPR 256 / private 288 B / LDS 25272). Measured alongside the SHRINK=0 row
(4ba4c94 source, lib `8816eebf589b0ebf`, VGPR 251 / private 32 B): binaries differ in hash and ISA
(fenced build carries 210 explicit `s_waitcnt_vscnt` waits; SHRINK=0 has none).

## (1) Correctness FIRST — determinism evidence (standing rec-pp2 battery, all cotenant-gated windows)

- **wrap512, fenced build, 2 fresh sessions × 2 KV modes**: fp32kv `dabb2919bbc8` ×2, bf16kv
  `eec63518fa18` ×2 — **byte-identical to the rec-3cr SHRINK=0 lop8 remedy md5s** (deterministic and
  output-equivalent to the pinned build). Window `fut11b-20261123T103209Z` (stageC).
- **depth8192 fp32kv, fenced build, 3 fresh sessions identical**: `c142fc5397fb` in the rec04i window +
  ×2 in stageD (`840260f`). Plus the cross-build point: fenced default-build depth8192 fp32kv session-1
  md5 `8e8c01246c73` = the rec-kcb SHRINK=0 default baseline's 2-session md5 — 3 independent sessions
  across two codegens agree byte-for-byte (the rec-3cr race signature was different output per codegen).
- **Encoding probes**: fut11b_probes.hip 4 probes × 2000 reps × 3 runs, all bit-exact vs CPU reference.
- **O-proj K=4096 row-0 edge**: the fence adds explicit `vmcnt(0)` + `s_waitcnt_vscnt null,0x0` at both
  `q4_rows_tail` call sites — the exact corner sunny memo 6dbb2bc identified (caller held `vmcnt(2)`
  across the call; callee entry waited vmcnt/expcnt/lgkmcnt but not vscnt).

Determinism verdict: **CLEAN** — meets and exceeds the 3×-identical spec; SHRINK=0 pin removed at HEAD.

## (2) Binary differs — verified

`ec34a71153bae74b` (fenced, 210 explicit vscnt waits) vs `8816eebf589b0ebf` (SHRINK=0); resources
VGPR 256/scratch 288 B vs 251/32 B; disasm `+12` vmcnt0+vscnt fence pairs at the call sites.

## (3) Timing — depth8192 ABBA 12/12 (prod bf16-KV, depth 8215..8265, k=3, seed 8192, same protocol as the 32.73 row)

| variant | lib | tok/s @8192 (12/12) | source |
|---|---|---|---|
| SHRINK=0 (pinned recipe) | 8816eebf | 32.73 | e5d window 2026-11-23T05:09Z |
| **SHRINK=1 + fence** | ec34a711 | **33.42** | rec04i window 2026-11-23T11:31Z |
| **SHRINK=1 + fence (replicate)** | ec34a711 | **33.38** | rec04irep 2026-11-24T00:23Z |

## Promotion (resolves on data)

Fenced variant replicate-confirmed faster (33.42 / 33.38, mean 33.40 vs 32.73, +2.0%) **with clean
determinism** (correctness-first evidence above) and is the HEAD default → **promote SHRINK=1+fence as
the full-config lop8 recipe**; pin updated in `docs/gates/rec-kcb-full-config-tiers-v1.md` +
`qgate/gate_spec_full_v1.json` (the `-DGEPA_SHRINK=0` recipe remains only for reproducing the measured
8816eebf lib). Caveats carried: rows are prod bf16-KV mode (rec-kcb ruling makes fp32-KV the
run-of-record — fp32-KV production row re-measure pending the next Alex window, expected below 33.4 at
8192 since fp32 KV doubles KV bytes read at depth); rec-cnk strict-tier depth logit FAILs are the known
router-tie artifact, disclosed.
