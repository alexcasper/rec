# rec-e5d iter 2 — full-config confirmation @8192 on dusk (2026-11-23, coordinator-approved window)

Source `390c4cd` (default-off knobs), built `-DFLINT_CFG_FULL=1` (forces GEPA_SHRINK=0):
`full_lop4.so` (`GEPA_ATTN_SPLIT=1 GEPA_ATTN_LOP=4`, sha256 d01d28a4c273c2cf…, VGPR 245, LDS 25272) and
`full_gqa.so` (`GEPA_ATTN_GQA=1`, sha256 430b6f260ff6bc14…, VGPR 247, LDS 25272). Script `scripts/m3_e5d_full8k_dusk.sh`.

## Window (cotenant state per line in `window.log`)
- 01:02:01Z pre-check: busy 0/0/0, MemAvailable 16.6 GiB, 4 llama-servers, :8080/:8090 200, llama-qwen inactive, display active.
- **01:02:10Z :8080 paused by ops_flint** (w11:p1Q). It confirmed in-pane and armed its own 30-min auto-restart watcher.
- 01:07:26Z run start (VRAM free 38.7 GiB). The 5-min gap was spent waiting for ops_flint's explicit confirmation, per the condition.
  A dusk-side watchdog was armed to abort at 01:22:00Z (20-min cap from the pause); it never fired.
- 01:07:26–01:12:07Z lop4 bench · 01:12:07–01:17:05Z gqa bench · 01:17:05–01:21:25Z lop4 fp32-KV emit. All rc 0; longest launch 0.192 s.
- 01:21:25Z DONE signalled · **01:22:52Z :8080 restarted** (health OK ~9 s after launch; ops_flint in-band depth512 29.48 tok/s, prefill
  440.18). Health verified from dawn at 01:22:54Z. **Total :8080 downtime ~20.7 min**, about 0.7 min over the ~20-min cap. The overrun came
  from the pre-start confirmation wait; GPU work took 14.0 min.

## Rows (production bf16-KV, depth8192 seeded prompt, ABBA 12 pairs after the emit, depth 8215..8265, k=3)
| build | decode @8192 | vs incumbent 26.80 | vs split-K 21.55 (gate8k) | prefill 8192 |
|---|---|---|---|---|
| **lop4** | **31.36 tok/s (12/12)** | **1.170×** | 1.455× | 278.4 s (32.4 ms/pos) |
| gqa | 29.53 tok/s (12/12) | 1.102× | 1.370× | 294.9 s (35.5 ms/pos) |
Both clear the ≥1.0× ungated success bar. The mid-based projection was lop4 ≈31.4 / gqa ≈30.3.

## fp32-KV gate, lop4 @8192 (dawn torch fp32 CPU reference)
| measure | value |
|---|---|
| tokens | 24/24 exact (same as the default and split-K builds) |
| strict spec (a) as written | **FAIL**: last 7.412e-2 / step 0.2969, 3 routing deviations (identical to default/split; the unforced reference trajectory) |
| routing deviations | 3, all 1-ulp-class k/(k+1) exchanges: L3@3232 gap 4.77e-7, L26@3255 4.47e-7, L38@6456 2.09e-7 |
| forced to kernel routing (rec-kcb proposal, NOT enacted) | **max 1.93e-5 over 25 rows: PASS 1e-4** (default 1.7e-4, split-K 2.2e-4 earlier) |
| kernel vs kernel | vs split-K step0 4.8e-7 / max 3.7e-6; vs default 1.1e-4 / 1.7e-4 |

## Not measured in this window (flagged)
- gqa fp32-KV emit (only one emit was budgeted). On mid, gqa passed strict at 8192 (1.24e-5).
- Full-config fresh-session determinism for lop4/gqa: one session each. Mid determinism passed (suite×2, wrap512×3 KV, 8192×2).
- Full-config wrap512 / ppl for the new builds: mid gates passed.

## Verdict
lop4 is the measured M3 depth build: 31.36 tok/s @8192 = 1.170× incumbent, tokens exact, forced fp32-KV 1.93e-5. Strict fp32-KV fails only
through the router ties already characterised in rec-cnk. Promotion of the full-config depth row routes to Alex under rec-kcb.
