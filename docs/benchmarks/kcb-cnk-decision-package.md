# Decision package for Alex — two open gate items (rec-cnk tie tolerance, rec-kcb v1 ratification)

Prepared 2026-11-23, dusk-free (docs lane). Sources: rec-cnk profile record `8333bcb`/`ca62e27`
(docs/benchmarks/rec-cnk-profile-2026-11-22/), rec-e5d iter-2 record `3d07fdb`/`3da8186`
(docs/benchmarks/e5d-iter2-mid-dusk-2026-11-22/), rec-ohs ledger `386921a`
(docs/benchmarks/rec-ohs-ledger-2026-11-22.md), gate spec `55b7840`
(docs/gates/rec-kcb-full-config-tiers-v1.md). Confirmation numbers for the live rec-e5d window
(DOWN 01:02:10Z, 30-min timeout) slot in when DONE lands; nothing here depends on it.

---

## Item 1 — rec-cnk: tie-tolerance amendment to the fp32-KV tier

### Evidence chain (all recorded, kernel exonerated)

1. **The strict-tier depth-8192 failure is one fp32 routing tie, in BOTH builds.**
   Full config: default 7.410e-2, split-K 7.412e-2 vs tier 1e-4 — kernel-vs-kernel agreement
   ~1e-4, tokens identical, determinism PASS. Mid-8L @8192 default passes strict at 1.24e-5,
   so it is full-config + depth, not a global defect.
2. **The tie, precisely:** full config = layer 3, position 3232, k/(k+1) exchange at
   selection-score gap 4.5–4.8e-7 (2^-21, one fp32 ulp); 1 deviation in 386k router decisions
   (8216 positions × 47 MoE layers). The rec-e5d mid @8192 builds show the same class:
   layer 6, position 7288, gap 6.6e-7 — gqa/def0 break it the torch way, split0/lop4 don't.
3. **Kernel exonerated against both reference paths** (routing forced to the kernel's emitted
   experts, 25 rows @8192): kernel vs stock reference 1.1e-6 step-0 (max 1.7e-4 on rows 15/21,
   top-2 margins 4.6/7.3, argmax equal); kernel vs chunked-prefill reference 1.1e-6 / 2.2e-4;
   reference-vs-itself across the two paths 3.8e-7 / 4.8e-5. Neither reference is wrong; the
   kernel agrees with both.
4. **The unforced 2.5e-2 @4096 / 7.4e-2 @8192 is reference-trajectory change, not kernel
   error:** after the tie both builds give identical unforced numbers (2.476e-2 @4096); strict
   holds up to 3072 in both builds (4.3e-6 / 5.0e-6); onset = 3232 exactly.
5. **Frequency argument:** ~47 router decisions per position means an exact fp32 tie (1 ulp)
   appears within a few thousand positions. A zero-deviation fp32-KV rule fails correct
   kernels at depth by construction.
6. Not RoPE convention (kernel-style exp2f inv_freq reference gives the same 7.415e-2); not a
   partial-buffer overflow (n_split ≤ 16).

### Proposed gate wording (for enactment in the pool hard_constraints + gate tool)

> **fp32-KV tier, full Flint config (amendment to spec (a) / rec-kcb v1):**
> (a) *Routing:* the reference is forced to the kernel's emitted per-position expert choices
> (`step_experts`). A kernel routing deviation is tolerated iff it is a k/(k+1) exchange
> (adjacent ranks) at selection-score gap ≤ 2e-6 (~4 fp32 ulps; observed ties 4.5–6.6e-7).
> Any routing deviation that is not an adjacent-rank exchange, or whose gap exceeds 2e-6, fails.
> (b) *Logits ≤ 4096 depth:* kernel vs forced reference, max row rel ≤ 1e-4.
> (c) *Depth-8192 allowance:* forced rows may reach max rel ≤ 2e-4, against measured reference
> reorder noise of 4.8e-5 at that depth (observed kernel forced rows: ≤2.8e-5 up to 4096;
> 1.3–1.7e-4 on 2/25 rows @8192). Alternative if the allowance is declined: at 8192 the binding
> check is ppl on eval_set_full_v1 (+ determinism) only, logit rel recorded informational.
> (d) *Unchanged:* N-repeat determinism (≥2 fresh-session emits byte-identical, all KV modes);
> ppl |Δ| < 1% on eval_set_full_v1; bf16-KV 1e-3 logit tier informational at full config,
> binding at mid; production bf16-KV |rel| < 1% vs reference.
> The tie-tolerance applies at mid config too (the L6@7288 case), where it is currently moot
> because default passes strict.

**Ruling options:** (i) ratify as worded; (ii) tighten the tie gap to 1e-6 (2 ulps — still
covers both observed cases, less headroom); (iii) ratify (a)/(b) but decline the 8192
allowance (c), binding on ppl+determinism at 8192; (iv) reject — then 8192 rows stay
unquotable for any non-bit-exact kernel and the depth front stays llama.cpp.

---

## Item 2 — rec-kcb v1 ratification + the two concrete depth-build acceptance cases

### v1 summary (committed `55b7840`, interim pending Alex)

- **bf16-KV 1e-3 logit tier is unattainable at full Flint** by any non-bit-exact kernel:
  reference-vs-reference (torch fp32-KV vs torch bf16-KV) diverges 1.85e-2 at step 0 (~18× the
  tier) with an argmax flip at step 3; kernel shows the identical gap; confirmed a BASELINE
  property (default 4.41e-2 ≈ split 2.74e-2 @512; production bf16kv ~6e-3 in all builds incl.
  def0, candidates within 1.5×).
- **Binding at full config:** fp32-KV ≤1e-4 (amended per Item 1), N-repeat determinism, ppl
  <1% on eval_set_full_v1 (8 real-text × 192 tok, sha 1b06d928). bf16-KV 1e-3 stays BINDING at
  mid (all current builds pass at 1.8–1.9e-4).
- Ratification makes the gated-512 row quotable (33.55 tok/s vs 29.39 = **1.14×**, fp32kv
  8.3e-7, determinism 3/3 modes, ppl PASS) and retires the R11r 0.84× default-build row to
  "default build" status.

### The two depth-build acceptance cases (rec-e5d iter-2, mid @8192 measured; full projected)

| | **lop4** (lanes-over-positions) | **gqa** (warp per kv-head × chunk) |
|---|---|---|
| mid @8192 speed | 154.9 tok/s = 1.61× split0 | 144.1 = 1.50× split0 |
| natural strict 8192 fp32kv | FAIL — the single 1-ulp tie (L6@7288, 6.6e-7) | **PASS 1.24e-5** |
| forced 8192 fp32kv (under Item 1) | **1.26e-5 PASS** | 1.24e-5 PASS |
| determinism / G0 / wrap512 / ppl | all PASS (ppl ≤6.9e-4) | all PASS |
| projected full @8192 (unmeasured) | **~31.4 tok/s = 1.17×** vs 26.80 | ~30.3 = 1.13× |
| short-ctx vs 2b98737 | 0.970/0.971 (vs def0 −0.5/−1.0%) | 0.966/0.962 (vs def0 −0.9/−1.9%) |

Short-context note: all four SHRINK=0 builds fail strict non-regression vs 2b98737, mostly
because of the SHRINK=0 pin itself (def0 alone is −2.5/−1.9%); the mid default (2b98737,
SHRINK=1, split off) is unchanged — these are full-config depth builds, not mid promotions.

### The decision, concretely

- **Ratify v1 + Item 1 (i or ii)** → lop4 is the M3 depth build; its owner-window confirmation
  (running now, DONE by ~01:32Z) either lands full @8192 ≈ 31.4 = 1.17× (first gated megakernel
  win at depth) or the projection is corrected; gqa stays the strict-clean fallback if the
  window contradicts the projection.
- **Ratify v1 but decline the tie allowance (iii)** → gqa becomes the depth build (passes
  natural strict, 1.13× projected); lop4/split0 rows stay unquotable at 8192.
- **Decline v1** → full-config rows stay gated as today: 512 win unquotable, 8192 unmeasurable,
  and the adopt-llama.cpp fallback becomes the default answer at the next checkpoint.

One flag, for honesty: the tie allowance is the piece that makes depth rows pass, and it is
justified by "one ulp on a degenerate trajectory, kernel exonerated against two reference
paths" — solid, but it is still a gate relaxation. Declining it costs the 1.17× depth
projection and keeps 1.13× (gqa); nothing else in the matrix changes.

---

## Addendum (2026-11-23 03:20Z — rec-e5d gap closure + iteration 3; gemini-kernels `4ba4c94`, pool updated)

**The draft 2e-6 tie band is now known to be too tight — and possibly unnecessary.** Two new facts:

1. **gqa's own full-config fp32-KV emit @8192** (tokens 24/24 exact, strict 1.215e-4/5.24e-4): **5 k/k+1 ties, two at 3.28e-6 and 2.53e-6 — ABOVE the draft 2e-6 band.** gqa is not tie-free at full config; its earlier "strict-clean" status was mid-8192 only. Forced scoring passes at 9.5e-6. So under the draft amendment, gqa would FAIL too — no build is clean under a 2e-6 band.
2. **Iteration 3 found lop8** (8 positions per warp iteration): mid ctx8192 **166.82 tok/s (+7.9% over lop4's 154.6)**, ALL mid gates PASS — including **8192 fp32kv STRICT PASS 1.25e-5 with ZERO routing deviations** — determinism, G0 8/8, ppl PASS, short 0.978/0.981 (same SHRINK=0 pin class as def0). Projected full @8192 **~33.0 tok/s (~1.23× vs 26.80)**, UNMEASURED, needs an owner window.

**Revised ruling options** (supersedes the Item-1 option list; evidence unchanged for the exonerations):
- **(i) Widen the tie band to 4e-6 (~8 fp32 ulps)** — covers every observed tie (4.5–6.6e-7 kernel-side, 2.5–3.3e-6 gqa-side), keeps forced-judging and the ≤1e-4/2e-4 logit tiers as drafted. All builds then quotable; risk: a wider band admits marginally larger router flips (still adjacent-rank only).
- **(ii) Keep 2e-6 and gate the depth row on lop8**, which ran tie-free at mid 8192 — if its full-config emit is also 0-dev, no allowance is consumed for the shipped build; the allowance exists only to make other builds' historical rows quotable. Risk: lop8's full-config tie behavior is unmeasured until its owner window; if ties appear above 2e-6, the row fails and a re-ruling is needed.
- **(iii) as drafted (2e-6)** — quotable builds: none at 8192 (lop4/split0/gqa all carry ties above band); depth stays incumbent territory pending lop8.
- **(iv) depth allowance only, no tie band** — 8192 judged on forced logits (≤2e-4) + ppl + determinism, routing ties recorded informationally. Simplest; weakest router guarantee.

Recommendation: **(ii) with (i) as fallback** — measure lop8 at full depth first (one ~12-min owner window, same shape as the last three); its strict-clean mid record makes it the only candidate that might need no relaxation at all, which is the strongest possible position for the decision matrix. If lop8 shows ties at depth, fall back to (i).

Speed table update (full @8192, vs incumbent 26.80): measured lop4 **31.36 (1.17×)**, gqa 29.53 (1.10×); projected lop8 ~33.0 (1.23×, unmeasured). Mid ctx8192: lop8 166.82, lop4 154.6, split0 96.4, def0 30.09. All commits local (`4ba4c94` gemini-kernels; pool updated); push ledger unchanged.

## Addendum 2 (2026-11-23 06:39Z — lop8 full-config @8192 CONFIRMED; gemini-kernels `33a9d42`, docs/benchmarks/e5d-lop8-full8k-dusk-2026-11-23/)

**lop8 measured: 32.73 tok/s (12/12 stable ABBA pairs) = 1.221× vs incumbent 26.80** — the
projection (~33.0) held within 0.8%. Window: 04:09:30–04:23:14Z, 4/4 rc 0, longest launch under
cap, 14m37s downtime.

Gates under the interim tiers: wrap512 fp32kv PASS 7.64e-7; fresh-session determinism PASS
(wrap512 fp32kv+bf16kv ×2, identical md5s); ppl eval_set_full_v1 PASS (fp32kv +1.2e-7;
bf16kv +6.92e-3 <1e-2, thinnest margin 3.1e-3). **8192 fp32kv strict still FAILs (7.4115e-2,
tokens exact) via 3 k/k+1 ties — and all three fall WITHIN the drafted 2e-6 band:**
L3@3232 4.77e-7, L26@3255 4.47e-7, L38@6456 exact 0.0-gap fp32 tie. Forced to kernel routing:
2.14e-5 PASS. The tie population is identical to lop4/default/split — baseline, not build.

**Consequence for the ruling: option (ii) is now executable AS DRAFTED — no band widening
needed.** Ratify rec-kcb v1 + the Item-1 amendment (2e-6 forced-judging), and lop8's 8192 row
is quotable at **1.22× gated** (7.41e-2 becomes informational under forced judging; the
unforced number is recorded either way). Without ratification, the 8192 row stays unquotable
and the depth front remains incumbent territory on paper despite the measured 1.22×.

Full-config decision matrix after this window (vs incumbent 26.80 @8192 / 29.39 @512):
| depth | build | tok/s | ratio | gating status |
|---|---|---|---|---|
| 512 | split-K + SHRINK=0 | 33.55 | **1.14×** | quotable under rec-kcb v1 |
| 8192 | **lop8** | **32.73** | **1.22×** | needs ONLY the Item-1 amendment (all 3 ties ≤2e-6) |
| 8192 | lop4 | 31.36 | 1.17× | needs amendment + band widened to 4e-6 (3 ties, 2 above 2e-6) |
| 8192 | gqa | 29.53 | 1.10× | needs amendment (5 ties, 2 above 2e-6) |

Mid front unchanged: 2b98737 214.27 short-ctx (byte-identical); lop8 mid ctx8192 166.82.
Recommendation stands: ratify v1 + Item-1 amendment as drafted; ship lop8 as the M3 depth
build. The megakernel then holds gated wins at BOTH decision depths — 1.14× @512 and 1.22×
@8192 — with footprint 21.0 GiB vs 18.5 GB (bytes/token disclosed per row) and prefill
remaining the known open gap (37 vs 450 pos/s serial; batched prefill queued).

## Addendum 3 (2026-11-23 11:40Z — lop8f CONFIRMED; supersedes lop8 as the recommended depth build)

**lop8f** (lop8 + SHRINK=1 re-enabled + the rec-11b fence, sha ec34a711) measured in the
rec04i window (4/4 rc 0, longest launch 0.191 s, 34.6 min downtime):

- **depth8192: 33.42 tok/s (12/12 stable ABBA pairs) = 1.247× vs incumbent 26.80** — front-best,
  supersedes lop8's 32.73. Projection track: 33.0 → 32.73 (lop8) → 33.42 (lop8f).
- wrap512 determinism PASS both KV modes ×2 fresh, md5s (fp32kv dabb2919, bf16kv eec63518)
  **identical to the SHRINK=0 pin's outputs** — the fence re-enables SHRINK output-neutrally.
- speed512 on the same build: 25.04 tok/s vs R11r 24.75 (+1.2%).
- ppl eval_set_full_v1 bf16kv 3.2987 (matches lop8); fp32kv emit @8192 recorded (md5 c142fc53),
  dawn scoring (strict/forced + tie census) is the one formal step open.
- Residual: 8192 fp32kv ×2-fresh determinism formalization folds into the next window
  (in-window emit md5 8e8c0124 matches the historical deterministic default-build md5 —
  evidence green, pair formalization pending).

**Single-build consequence:** with SHRINK re-enabled, lop8f covers BOTH decision depths —
no separate 512/8192 builds. The @512 0.84× shrunk-build row (R11r) is obsolete as a
build-selection problem; a re-baselined @512 row on lop8f (one short window) would replace it.

**The ruling Alex owes is unchanged and now settles more:** ratify rec-kcb v1 + the rec-cnk
amendment (2e-6 forced-judging; lop8f's tie census lands with its fp32kv scoring) → the
megakernel ships gated at **1.14× @512 (33.55, split-K+SHRINK=0) and 1.25× @8192 (33.42,
lop8f)** vs the incumbent's 29.39/26.80 — or adopt llama.cpp for the depth rows on paper while
the measured wins go unquoted. Footnote as always: footprint 21.0 GiB vs 18.5 GB, prefill
37 vs 450 pos/s (batched prefill queued).
