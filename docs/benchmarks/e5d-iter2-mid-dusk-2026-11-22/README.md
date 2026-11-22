# rec-e5d iteration 2 — depth-attention redesign, mid-8L on dusk (2026-11-22, 22:58–23:47Z)

Source `390c4cd` (gemini-kernels). Knobs are default-off. Default, split, full and full+split builds produce the same instruction streams as
HEAD (only the `__hip_cuid` source-hash symbol differs). Every build uses `-DGEPA_SHRINK=0` (the full-config pin), on mid-8L (mid-prod bank).
:8080 stayed up; no owner window. Cotenants were idle on every pair (busy 0 at start; flint/E4B idle, llama-qwen inactive; logged per row).

| build | knobs |
|---|---|
| def0 | split off (default attention) |
| split0 | `GEPA_ATTN_SPLIT=1` (split-K as built, MIN 64, n_split <= 16) |
| **lop4** | `GEPA_ATTN_SPLIT=1 GEPA_ATTN_LOP=4`: candidate (1) lanes-over-positions, 4 positions per warp iteration, 8 lanes × 16 dims, 3-step reduction |
| **gqa** | `GEPA_ATTN_GQA=1`: candidate (2) one warp per (kv-head, chunk) serves all 8 q-heads, so K/V is read once per kv-head; up to 32 chunks; sink-aware merge |

## Speed (rehearsal window-bench, 12/12 stable ABBA pairs per row, `bench_summary.log`)
| build | ctx8192 tok/s | vs split0 | short (depth 4..328) |
|---|---|---|---|
| def0 | 30.09 | 0.31× | 184.9 |
| split0 | 96.4 / 96.6 / 96.4 / 96.3 | 1.00× | 204.6 / 203.2 / 204.0 / 203.9 |
| gqa | 144.22 / 144.02 | **1.495×** | 199.6 / 200.1 |
| lop4 | 154.76 / 154.95 | **1.606×** | 207.3 / 207.5 |
Run order: split0 gqa gqa split0 lop4 split0 split0 lop4 def0. Longest launch 0.198 s.

**Full-config projection (not measured).** Mid has 4 full-attention layers; full Flint has 14 (both 32 q / 4 kv heads, window 513). The
split-K depth-8192 overhead per full-attention layer is 1.37 ms on mid ((1/96.4 − 1/204) s ÷ 4) and 1.44 ms on full ((1/21.55 − 1/38.1) s ÷ 14),
so it transfers per layer. Per layer: lop4 0.41 ms, gqa 0.49 ms. Projected full @8192: **lop4 ≈ 31.4 tok/s (1.17× vs 26.80), gqa ≈ 30.3
(1.13×)**. This needs an owner window to confirm. The success bar (≥1.0× ungated at full 8192) is projected cleared, not measured.

## Gates (dawn torch fp32 CPU reference; strict = spec (a) as written; forced = reference forced to the kernel's experts, rec-kcb proposal, NOT enacted)
| build | determinism (suite×2, wrap512×3 KV, 8192 fp32kv×2 fresh) | G0 vs 24f167d | wrap512 fp32kv | wrap512 fp32_bf16kv (mid-binding 1e-3) | wrap512 bf16kv (prod) | 8192 fp32kv strict | 8192 fp32kv forced | ppl v1 fp32/bf16 | ppl full fp32/bf16 |
|---|---|---|---|---|---|---|---|---|---|
| def0 | PASS | 8/8 | 1.16e-6 | 1.82e-4 PASS | 5.91e-3 (forced 2.99e-3, 37 devs) | PASS 1.24e-5 | 1.24e-5 | −3.5e-8 / −6.9e-4 | +2.5e-8 / +1.6e-4 |
| split0 | PASS | 8/8 | 9.50e-7 | 1.94e-4 PASS | 6.68e-3 (3.01e-3, 37) | FAIL: 1 dev L6@7288 gap 6.6e-7 | 1.25e-5 | same | +1.8e-8 / +2.0e-4 |
| **lop4** | PASS | 8/8 | 1.02e-6 | 1.91e-4 PASS | 5.75e-3 (2.91e-3, 41) | FAIL: 1 dev L6@7288 gap 6.6e-7 | **1.26e-5 PASS** | same | +2.5e-8 / +1.7e-4 |
| **gqa** | PASS | 8/8 | 9.94e-7 | 1.87e-4 PASS | 5.93e-3 (2.98e-3, 38) | **PASS** 1.24e-5 | 1.24e-5 | same | +1.8e-8 / +2.0e-4 |
- The production bf16kv strict failure (~6e-3) appears in all four builds, including def0. That makes it a baseline property, and every candidate is within 1.5× of def0 (qgate T4 form).
- The 8192 strict failures (split0, lop4) are the same single fp32 1-ulp k/(k+1) router tie (L6@7288, gap 6.6e-7) seen in rec-cnk. gqa and
  def0 happen to break that tie the torch way. Forced, all four builds agree at 1.24–1.26e-5.
- ppl: all |Δ| ≤ 6.9e-4, far below 1%.

## Short-context non-regression vs run-of-record 2b98737 (gepa bench ABBA ×8, 0 drops; `*_gate.log`)
| build | short4 | chat |
|---|---|---|
| def0 | 0.9747 | 0.9807 |
| split0 | 0.9673 | 0.9733 |
| lop4 | 0.9703 | 0.9710 |
| gqa | 0.9658 | 0.9622 |
**All four fail strict non-regression against 2b98737.** Most of the gap is the `GEPA_SHRINK=0` pin: def0 itself is −2.5/−1.9%, and 2b98737
is SHRINK=1. Against def0 (same pin), lop4 is −0.5/−1.0% and gqa −0.9/−1.9%. These are full-config depth builds, where SHRINK=0 is forced.
They are not mid-default promotions, and the mid default (2b98737, SHRINK=1, split off) is unchanged.

## Verdict
Candidate (1) **lop4 is the provisional M3 depth build**: fastest (1.61× split-K at mid 8192), deterministic, forced fp32-KV 1.26e-5. It fails
strict 8192 only on the known 1-ulp tie. Candidate (2) gqa is the strict-clean alternative (1.50×). Candidate (3) (more splits) was not
needed. Promotion of the full-config depth row stays Alex-gated (rec-kcb), and the full-config speed claim needs an owner window.
