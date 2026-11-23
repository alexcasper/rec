# rec-ohs Hill-Climb Candidate Ledger (2026-11-22)

Consolidated candidate ledger for the GEPA hill-climb on megakernel decode (rec-ohs) and the
full-config rows it branched into (M3 → rec-3cr / rec-kcb / rec-cnk / rec-e5d).
Sources of truth: `bd show rec-ohs` comments (2026-11-22 00:33Z–23:58Z) and pool commits on
`rec/qwen-aip-prep` (29951f0…3da8186). Kernel-side records on `rec/gemini-kernels` (local, unpushed).
Dusk mid-8L short-ctx unless noted; ABBA ratios are candidate/24f167d paired deltas; acceptance
threshold +1.6% on both cases (noise 0.8%); gate = rec-pp2 two-tier vs torch fp32 CPU
(fp32kv ≤1e-4, fp32_bf16kv ≤1e-3) + ppl |Δ|<1% on frozen eval_set_v1, per the 01:03Z ruling.

**Byte-parity column key**: `byte-id` = outputs bit-identical to baseline (gates inherited);
`relaxed` = arithmetic or reduction order legally changed (gate margins shown); `codegen-pin` =
remedy build (SHRINK=0/force-inline), parity vs 2fd83cb lineage n/a; `race` = known-nondeterministic.

## Ledger

| id | config | lever | result (tok/s or ABBA ratio s4/chat) | gates fp32kv / bf16kv | byte-parity | verdict + reason |
|---|---|---|---|---|---|---|
| iter0 | mid s4/chat | no mutation (incumbent vs copy) | 207.73/207.18 | inherit | byte-id | baseline reproduction; noise 0.8% |
| c001 | mid | unroll 2 | 0.976/0.979 | inherit | byte-id | REJECT slow |
| c002 | mid | unroll 4 | 0.873/0.876 | inherit | byte-id | REJECT slow |
| c003 | mid | K512 pipelined | 1.015/1.009 | inherit | byte-id | REJECT below +1.6% |
| c004 | mid | no x-prefetch | 1.006/1.000 | inherit | byte-id | REJECT below threshold |
| c005 | mid | K512_PIPE + no x-prefetch (kimi) | 1.011/1.018 | inherit | byte-id | REJECT short4 below threshold |
| c006 | mid | workers 8 | 0.932/0.936 | inherit | byte-id | REJECT slow |
| c007 | mid | per-WG top-K | 0.998/0.991 | inherit | byte-id | REJECT |
| c008 | mid | residual fusion | 1.002/0.993 | inherit | byte-id | REJECT |
| c009 | mid | s2-moe-grp (g64 group-to-lane) | 0.967/0.967, ppl +6.5e-4 | pass | relaxed (load issue) | REJECT slow; Step 2 negative |
| c010 | mid | s2-moe-half | 0.949/0.957 | pass | relaxed | REJECT slow |
| c011 | mid | s2-moe-ldspack (literal LDS staging) | 0.959/0.967 | pass | relaxed | REJECT; LDS hop costs (confirms rec-6ob.1) |
| c012 | mid | s2-lmh-grp | 0.999/1.007 | pass | relaxed | REJECT; stop rule hit |
| c013 | mid | s2-all-grp (I-cache control) | 1.004/1.003, hot 11582<11911 | pass | relaxed | REJECT neutral: K=2048 coalescing does not bound decode |
| c014 | mid | s2-all-half | 0.982/0.979 | pass | relaxed | REJECT slow |
| c015 | mid | d1-pf4k (idle-warp L2 prefetch) | 1.007/1.006 | inherit | byte-id | REJECT <1% |
| c016 | mid | d1-pf8k | 0.991/0.989 | inherit | byte-id | REJECT slow |
| c017 | mid | d1-pf2k | 1.004/1.002 | inherit | byte-id | REJECT <1% |
| c018 | mid | d1-pf16k | 0.961/0.969 | inherit | byte-id | REJECT; long prefetch delays barrier |
| c019 | mid | actfold (grid barrier → syncthreads) | 0.989/1.002 | inherit | byte-id | REJECT |
| c020 | mid | actfold + pf4k | 0.992/1.001 | inherit | byte-id | REJECT |
| c021 | mid | actfold + residfuse (−15 barriers/tok) | 0.992/1.000 | inherit | byte-id | REJECT |
| c022 | mid | actfold + resid + pf4k | 0.990/1.000 | inherit | byte-id | REJECT |
| c023 | mid | c1-osplit O-proj split-K (runtime stride) | 189.8/188.3 = 0.917/0.907 | fp32kv pass; bf16kv 9.36e-4 HEADROOM FLAG (margin 6.4e-5); ppl +8.1e-4 | relaxed | REJECT; scratch spill 16→128 B hits shared K=2048 loop |
| c024 | mid | c1b-osplit-ldt (compile-time stride) | 198.0/199.3 = 0.957/0.963 | bf16kv 9.36e-4 FLAG | relaxed | REJECT below threshold; less depth + I-cache cost |
| c025 | mid | c2-downcat (MoE down = virtual K=2048) | 207.4/206.7 = 0.996/0.997 | bf16kv 9.31e-4 FLAG; vgpr 256 (cap) | relaxed | REJECT neutral |
| c026 | mid | K512_PIPE+DUAL+DIST2 (kimi) | 208.4/209.5 = 1.001/1.016 | inherit (bf16kv unchanged 9.115e-4) | byte-id | REJECT short4 < +1.6% |
| c027 | mid | K512_PIPE (c003 repeat) | 209.2/209.3 = 1.005/1.010 | inherit | byte-id | REJECT |
| c028 | mid | GEN_DIST2 (prefetch distance 2) | 207.2/206.9 = 0.998/0.998 | inherit | byte-id | REJECT |
| c029 | mid | K512_DUAL (2 quads/load round) | 208.1/209.5 = 0.994/1.010 | inherit | byte-id | REJECT |
| c030 | mid | Step 2 literal bulk-dequant-to-LDS (patch) | 197.5/197.4 = 0.951/0.949 | fp32kv pass; bf16kv 9.31e-4 FLAG; ppl −1.6e-4 | relaxed | REJECT; 'path to 260' refuted on dusk mid |
| c031 | mid | b6 barrier-cut | (reject) | inherit | byte-id | REJECT; flagged albion/R9700-window material |
| c032 | mid | act fused into down x-load | (reject) | inherit | byte-id | REJECT |
| **c033** | **mid** | **hot-code shrink (Track A)** | **214.27/214.32 = +3.05/+3.59% (replicate +2.65/+2.27)** | qgate 98/98 identical (verdict inherits) | **byte-id** | **PROMOTED → 2b98737 (new front)** |
| c034 | mid | combined | (reject) | — | byte-id | REJECT |
| M3 R11 (VOIDED) | full @512/@8192 | GEPA_ATTN_SPLIT=1 build | 34.40 @512 (1.19×); 21.35 @8192 (0.81×) | spot FAIL: rel 8.97e-3, step 2.73e-2 (rec-6ek) | race | VOIDED — racy build; superseded by R11r + gated rows; pool rejected[0] |
| R11r | full @512 / short | HEAD default (split off), rec-3cr re-run | 24.75 @512 (12/12); short 39.06 | fp32kv PASS 1.55e-6; determinism 8/8 | byte-id vs lineage | VERIFIED row: 0.84× vs 29.39 — megakernel loses at 512 on default build |
| r3c soak | full | codegen soak (control must race) | — | control SHRINK=1 3/3 distinct FAIL; SHRINK=0 8/8 identical PASS 1.55e-6; FORCE_INLINE 5/5; default+SHRINK=0 3/3 | codegen-pin | FIX 288f4fa: GEPA_SHRINK=0 default under FLINT_CFG_FULL |
| gated-512 | full @512 | split-K + SHRINK=0 | **33.55 (12/12) vs 29.39 = 1.14×**; short 38.1 | fp32kv PASS 8.33e-7; bf16kv FAIL 2.74e-2 (old tier) → ACCEPTABLE under rec-kcb v1 tiers (ppl PASS) | relaxed (split softmax merge) | ACCEPTED per-row under interim tiers; caveat: error grows with depth |
| gate8k split | full @8192 | split-K + SHRINK=0 | 21.55 (12/12) vs 26.80 = 0.80× | fp32kv FAIL 7.41e-2 (3 routing devs); determinism PASS | relaxed | REJECTED on binding tier; split-K exonerated later (rec-cnk) |
| default-8192 | full @8192 | default build (split off) | speed unmeasurable under 0.25 s cap (tA 245 ms; ≥4.08, ~7.9 implied) | fp32kv FAIL 7.410e-2 = split's 7.412e-2; kernel-vs-kernel ~1e-4; determinism PASS | byte-id lineage | REJECTED — depth failure is a BASELINE full-config property, not split-K |
| rec-cnk @3072 | full @3072 | default / split profile | — | strict PASS 4.3e-6 / 5.0e-6 | mixed | PASS — onset is above 3072 |
| rec-cnk @4096 | full @4096 | default / split profile | — | strict FAIL 2.48e-2 (1 dev = the tie); **forced** 2.8e-5 / 4.5e-6 PASS | mixed | ONSET = position 3232: one fp32-ulp k/(k+1) routing tie (gap 4.5–4.8e-7) |
| rec-cnk ref | full @8192 | forced-routing discriminators | — | kernel vs stock ref 1.1e-6; vs chunked ref 1.1e-6/2.2e-4; ref vs itself 3.8e-7/4.8e-5; 386k router decisions: 1 deviation | n/a | NO systematic depth error in kernel OR reference — the failure is one 1-ulp routing tie on a degenerate trajectory |
| e5d-i2 def0 | mid @8192 | default SHRINK=0 | 30.09 | 8192 fp32kv strict PASS 1.24e-5 | codegen-pin | strict-clean fallback |
| e5d-i2 split0 | mid @8192 | split-K SHRINK=0 | 96.4 (reps 96.3–96.6) | strict FAIL on 1-ulp tie; forced 1.26e-5 PASS | relaxed | depth build |
| e5d-i2 gqa | mid @8192 | one warp per kv-head × chunk | 144.1 = 1.50× | strict PASS | relaxed | strict-clean depth candidate |
| e5d-i2 lop4 | mid @8192 | 4 positions per warp iteration | 154.9 = 1.61× | strict FAIL (1-ulp tie L6@7288); forced PASS; ppl ≤6.9e-4 | relaxed | **provisional M3 depth build**; full @8192 projected ~31.4 = 1.17× (UNMEASURED, owner window pending) |

## Pareto front / current state

- **Mid-8L short-ctx front: `2b98737` (c033 hot-code shrink) = 214.27 tok/s**, byte-identical
  outputs (98/98 → T1–T4 verdict inherited), promoted from front `24f167d` (~208). Byte-identical
  ceiling was ~207–211 before the I-cache lever; the **"path to 260" is refuted on dusk mid**
  (c030 + all Step 2 variants ≤ front).
- **fp32_bf16kv headroom budget: ~6–7e-5 unspent** (builds sit at 9.1–9.4e-4 against the 1e-3
  tier); every relaxed candidate costs ~2e-5.
- **Full-config @512: split-K + SHRINK=0 = 33.55 tok/s vs 29.39 = 1.14× (gated win)** under the
  rec-kcb v1 interim tiers (fp32kv 8.3e-7, determinism 3/3 modes, ppl PASS).
- **Full-config @8192: blocked on rec-cnk, not on the kernel.** Both builds show 7.4e-2 strict
  fp32-KV drift, fully explained by ONE fp32-ulp routing tie (layer 3, position 3232, gap
  4.5–4.8e-7); forced-routing errors are 2.8e-5–4.5e-6 @4096 and kernel agrees with both
  reference paths (1.1e-6 @8192). Split-K is further exonerated (never worse than default;
  ~2.7× faster at depth). Speed at 8192: split-K 21.55 = 0.80× vs incumbent 26.80; rec-e5d iter-2
  lop4 projects **~31.4 = 1.17×** (unmeasured, owner window pending).
- **Pending rulings/items**: rec-kcb v1 tiers (interim, pending Alex); rec-cnk gate proposal
  (allow declared k/(k+1) exchanges at gap ≤2e-6, judge fp32-KV forced; possible depth allowance
  at 8192 where forced rows reach 1.3–1.7e-4 vs reference reorder noise 4.8e-5); rec-e5d owner
  window for full @8192 lop4/gqa ABBA; mid bf16-KV absolute-tier baseline issue (all builds
  ~6e-3, ratio-gated <1.5×).
- **Rejected[] ledger**: pool rejected[0] = GEPA_ATTN_SPLIT=1 (full-config codegen race,
  rec-6ek/rec-3cr); determinism gate (second emit byte-identical) now auto-rejects such
  candidates (`6a7d205`).
- All commits local; nothing pushed (standing rule: push on explicit ask).
