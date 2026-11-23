# rec-e5d gap closure — full-config gqa fp32-KV @8192 + fresh-session determinism (2026-11-23)

Coordinator-approved :8080-down window. A full-config session can't fit beside :8080: 18.0 GiB VRAM free with 4 llama-servers up, and the bank
needs 21.0 GiB plus 1.5 GiB slack. Libs as in the 01:07Z window: full_lop4.so (d01d28a4…), full_gqa.so (430b6f26…), source 390c4cd.
Script `scripts/m3_e5d_gap_dusk.sh`.

**Window:** :8080 PAUSED 02:25:21Z (ops_flint confirmed) → run 02:25:40–02:34:00Z (3/3 rc 0, longest launch 0.174 s; watchdog at 02:45:00Z
unused) → RESTARTED 02:34:43Z. Health verified from dawn 02:34:47Z; ops_flint in-band 29.64 tok/s. **~9.4 min down.** Cotenants per line in
`window.log` (E4B/qwen3vl 200, llama-qwen inactive, display active).

## gqa fp32-KV @8192 (full config, dawn torch fp32 CPU reference)
| measure | gqa | lop4 (01:07Z window) |
|---|---|---|
| tokens | 24/24 exact (identical to lop4) | 24/24 exact |
| strict spec (a) | FAIL 1.215e-4 / 5.24e-4 | FAIL 7.412e-2 / 0.2969 |
| routing deviations (all k/k+1) | 5: L14@7063 **3.28e-6**, L25@5899 **2.53e-6**, L30@4881 6.3e-7, L23@7515 5.8e-7, L31@7866 6.0e-8 | 3: gaps 2.1–4.8e-7 |
| forced to kernel routing (proposal, NOT enacted) | **9.5e-6 PASS** | 1.93e-5 PASS |
gqa breaks the L3@3232 tie the torch way, so its unforced reference stays close (1.2e-4). **Two of its tie gaps (3.28e-6, 2.53e-6) exceed the
2e-6 tolerance in the draft rec-kcb amendment.** Under that draft, gqa would fail. The ruling needs either a wider tie band (~4e-6) or a
relative-to-score-scale definition.

## Fresh-session determinism, full config (wrap512, 2 sessions each)
| build | fp32kv | bf16kv | wrap512 fp32kv vs torch | wrap512 bf16kv (prod, informational at full) |
|---|---|---|---|---|
| lop4 | PASS (md5 7579906f) | PASS (ee87c6fe) | PASS 7.6e-7 / 2.4e-6, 0 devs | FAIL 5.82e-2 (baseline property; split-K 4.03e-2, default 4.41e-2 fp32_bf16kv) |
| gqa | PASS (5e1029b9) | PASS (332ca2be) | PASS 7.0e-7 / 1.6e-6, 0 devs | FAIL 5.07e-2 (same) |
