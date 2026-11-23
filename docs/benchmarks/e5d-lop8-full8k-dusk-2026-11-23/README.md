# rec-e5d lop8 — full-config confirmation @8192 + binding-tier gates (2026-11-23, coordinator-approved window)

`full_lop8.so`: source 4ba4c94, `-DFLINT_CFG_FULL=1 -DGEPA_ATTN_SPLIT=1 -DGEPA_ATTN_LOP=8` (GEPA_SHRINK=0 forced), sha256 8816eebf589b0ebf…,
VGPR 251, LDS 25272. Class-B scan has the same profile as the soaked build (b=0, b'=12, c=12). Script `scripts/m3_e5d_lop8_dusk.sh`. Measurement only.

## Window (cotenant state per line in `window.log`)
- Pre-check 04:08:26Z: busy 0/0/0, MemAvailable 16.4 GiB, 4 llama-servers, :8080 200.
- **PAUSED 04:09:09Z** (ops_flint confirmed in-band) → run 04:09:30–04:23:14Z, 4/4 steps rc 0, longest launch ≤0.195 s; watchdog set for
  04:29:00Z, unused → DONE signalled 04:23:14Z → **RESTARTED 04:23:46Z** (ops_flint; dawn health 200 at 04:23:44Z, ~2 s clock offset).
- **:8080 downtime 14m37s** (04:09:09–04:23:46Z), under the 20-min cap. The ~6-min target covered the row alone; the binding gates added
  determinism and ppl emits.
- After restart, on-box cotenant gate (`scripts/bisect/inband_probe.sh`, dusk → tailscale IP): 04:24:04Z health 200, decode **30.16 / 30.10 tok/s**
  (64-token completions); ops_flint's own probe read 29.63 tok/s at depth 512. In band.

## Speed (production bf16-KV, depth8192 seeded prompt, ABBA 12/12 pairs, depth 8215..8265, k=3)
| row | decode @8192 | lop8 relative |
|---|---|---|
| **lop8 (measured)** | **32.73 tok/s** | — |
| projection from mid (iter 3) | 33.0 | −0.8% vs projection |
| lop4 (01:07Z window) | 31.36 | **+4.4%** |
| incumbent llama.cpp UD-Q3_K_XL @8192 | 26.80 | **1.221×** |
| incumbent solo baseline 31.1 (provenance held by ops_flint) | 31.1 | 1.052× |
| split-K as built (gate8k) | 21.55 | 1.519× |
Prefill 8192 in this window: about 288 s (row step).

## Binding full-config tiers (rec-kcb interim, pending Alex), torch fp32 CPU reference on dawn
| tier | result |
|---|---|
| fp32-KV ≤1e-4 @wrap512 | **PASS** last 7.64e-7 / step 1.47e-6, 0 routing deviations |
| fp32-KV ≤1e-4 @8192, strict as written | **FAIL** last 7.4115e-2 / step 0.2969 (margin: 741× over on last). Tokens 24/24 exact. 3 routing deviations, all k/(k+1): L3@3232 gap 4.77e-7, L26@3255 4.47e-7, L38@6456 **0.0** (exact fp32 tie). Identical to lop4 / default / split-K @8192 |
| fp32-KV @8192, forced to kernel routing (rec-kcb proposal, NOT enacted) | **PASS** max 2.14e-5 over 25 rows, none >1e-4 |
| ≥2 fresh-session determinism | **PASS** wrap512 fp32kv (md5 dabb2919) and bf16kv (eec63518), 2 sessions each. 8192 single session (mid: 2 sessions PASS) |
| ppl \|Δ\| <1% on eval_set_full_v1 (items sha 1b06d928…) | **PASS** fp32kv +1.20e-7; bf16kv +6.92e-3 (split-K was +3.38e-3; per-item deltas −0.0066…+0.0117 nats, mixed sign; max\|Δlogp\| 2.44 vs 1.84). Margin left 3.1e-3 |
| info: production bf16kv @wrap512 | 5.48e-2 (baseline property; lop4 5.82e-2, gqa 5.07e-2) |

## Verdict
lop8 is the **front-best candidate** for the M3 depth build, pending Alex's rec-kcb ruling: 32.73 tok/s @8192 = 1.221× incumbent, the fastest
measured full-config depth row. It passes every binding tier except the strict 8192 fp32-KV reading. That reading fails identically for every
build tested (default, split-K, lop4) because of the same router ties, including one exact fp32 tie, so it isn't specific to lop8. Forced, lop8
passes at 2.14e-5. Flag: the bf16-KV ppl spend (+6.9e-3 of the 1e-2 budget) is about twice split-K's; mid lop8 was +4.0e-5, so this looks like
routing-flip noise, but it's the thinnest margin.

## Build-recipe note after rec-11b (98ce41d, 2026-11-23)
rec-11b re-enabled `GEPA_SHRINK=1` (plus the VMEM fence) as the full-config default. The row above was measured on the **SHRINK=0** build.
At HEAD, `-DFLINT_CFG_FULL=1 -DGEPA_ATTN_SPLIT=1 -DGEPA_ATTN_LOP=8 -DGEPA_SHRINK=0` is instruction-identical to the measured `full_lop8.so`
(0 differing lines). Without `-DGEPA_SHRINK=0`, HEAD's full lop8 is a different, unmeasured build that spills (VGPR 256, private 288 B vs
251 / 32 B). **The lop8 promotion recipe must pin `GEPA_SHRINK=0`, or the SHRINK=1+fence lop8 variant needs its own full-config row and gates.**

## Gate ruling (Claude arm, acceptance interpretation), 2026-11-23: lop8f depth8192 fp32-KV (0742101)
**(a) Strict FAIL stands. The forced-routing PASS is not acceptance.** rec-pp2 as written (and the rec-kcb interim full-config tiers)
judges fp32-KV ≤1e-4 on the unforced evaluation, with any routing deviation a FAIL. The tie-band / forced-to-kernel-routing reading is a
rec-kcb *proposal* that is not enacted. lop8f @8192 fp32-KV is therefore **FAIL (7.411e-2 last / 0.2969 step)** for acceptance purposes.
Forced 2.14e-5 and the tie census (3 k/k+1 exchanges, gaps 4.77e-7 / 4.47e-7 / exact 0.0, all inside the draft 2e-6 band) are disclosed
as **evidence for Alex's ruling**, not as a pass. The @8192 speed row (33.42) is quotable only with this FAIL disclosed alongside it. Status
flips to PASS only if Alex enacts a tie band that covers these gaps.
**(b) Yes, the disclosure should mark it variant-independent (pre-existing vs the fp32 reference, not lop8f-introduced), worded precisely:**
every kernel build tested at depth 8192 on the seeded prompt fails the strict tier through the same router ties, with tokens 24/24 exact:
default 7.410e-2 (1 tie, L3@3232), split-K 7.412e-2, lop4 7.412e-2, lop8 7.4115e-2, lop8f 7.4115e-2 (3 ties at the same sites). gqa
breaks the L3@3232 tie the torch way and fails at only 1.215e-4 (5 ties, 2 above the draft band). The *occurrence* is variant-independent;
the *magnitude* depends on which side of the L3@3232 exact-scale tie a build lands, not on the attention variant's numerical quality
(forced: lop8f 2.14e-5 vs default 1.7e-4). Suggested disclosure line: "fp32-KV strict tier at depth 8192: FAIL for all kernel builds
(pre-existing, variant-independent router-tie divergence vs the fp32 reference; forced-routing 2.14e-5; acceptance pending rec-kcb)."
