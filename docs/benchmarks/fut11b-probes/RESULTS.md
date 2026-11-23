# rec-11b: ISA coverage probes + fenced-SHRINK determinism gate (2026-11-23)

## Fix (gemini-kernels 98ce41d)
GEPA_SHRINK=1 re-enabled at full config with `gepa_vmem_fence()` (explicit
`s_waitcnt vmcnt(0)` + `s_waitcnt_vscnt null, 0x0`) at both `q4_rows_tail` call
sites, FLINT_CFG_FULL-only (promoted mid c033 build untouched). The rec-3cr
SHRINK=0 pin is removed at HEAD.

## Race-window disasm evidence (stage A, s1 vs s0 k_gfx1150.s)
- Caller calls `q4_rows_tail` with `vmcnt(2)` deliberately outstanding ACROSS the call.
- Callee entry wait: `vmcnt(0) expcnt(0) lgkmcnt(0)` — no vscnt.
- Shrunk build scratch: 144 B (90 `scratch_store` / 111 `scratch_load`) vs 16 B pinned.
- Builds: fixed (+12 vmcnt0+vscnt fence pairs), control unfenced (183 vscnt, clean), mid fence-free.

## Stage B: encoding probes (fut11b_probes.hip, gfx1150)
4 probes x 2000 reps x 3 runs, :8080 up, busy-gated, cotenants logged:
P0 control / P1 same-type-WAW / P2 call-boundary spill churn / P3 shrink-replica —
ALL bit-exact vs CPU reference (`total_bad=0`). The class-B hazard is
layout-specific (rec-3cr: 3 codegens / 3 answers), hence a boundary fence,
not an idiom ban. Side note: hipcc rebuilds of the same logical config vary
in function order/RA (s1 vs midchk) — lane gates compare G0 outputs, unaffected.

## Stage C: determinism gate on the FIXED full build (m3window/fut11b-20261123T103209Z)
- Window: :8080 DOWN 10:33 -> followup end ~11:04Z (ops_flint p1Q); cotenants logged per step.
- wrap512 fp32kv: 2 fresh sessions, md5 8a8fcf6dc2a3... BOTH — DETERMINISM PASS
- wrap512 bf16kv: 2 fresh sessions, md5 d97d2aa67693... BOTH — DETERMINISM PASS
- depth8192 fp32kv session 1: md5 8e8c01246c73 — IDENTICAL to the rec-kcb default
  (SHRINK=0) 2-fresh-session md5 8e8c0124 => 3 independent fresh sessions across
  two codegens agree byte-for-byte. (The race signature was different md5s per codegen.)
- depth8192 session 2 refused: p1Q restarted :8080 mid-window; flint VRAM resident
  (18.36 GiB free < 23.3 needed). Gate refused correctly; not re-run.
- Speed @512 (fenced): 25.04 tok/s (12/12 stable pairs) vs incumbent 29.39 = 0.85x —
  fence is free (R11r shrunk row 24.75).

## Verdict
SHRINK=1 + boundary fence is deterministic at full config and byte-identical to the
pinned SHRINK=0 baseline. rec-3cr's pin is replaced by an explicit, targeted fence.
Strict-tier depth logit FAILs (7.4e-2) remain the known 1-ulp routing-tie artifact
(rec-cnk), gated separately by Alex's parked ruling.
