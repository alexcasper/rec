# rec-dor iteration 6 — mid-8L ctx8192 on dusk (2026-11-23), source de70bbe

Baseline ctl6 = `GEPA_ATTN_SPLIT=1 GEPA_ATTN_LOP=8` at mid (SHRINK=1), the mid equivalent of lop8f. Lever-off builds (default, split, ctl5b,
ctl5s0b, full, full lop8f) are instruction-identical to 5281d5c (0 differing lines) before any bench. :8080 up. Every row passed the extended
per-row cotenant gate (no foreign drivers, qwen3vl 0 ESTAB, busy sum <15 over 3 s, :8080/:8090 health logged).

| build | ctx8192 tok/s (12/12) | short (12/12) | verdict |
|---|---|---|---|
| ctl6 | 167.73 / 167.86 / 167.27 (167.62) | 208.02 / 207.93 / 208.14 | control |
| **dot2** `GEPA_ATTN_DOT2=1` | **173.52 / 173.50 (+3.5%)** | 209.30 / 209.82 (+0.7%) | live candidate (relaxed, production mode only) |
| smin16 `GEPA_ATTN_SPLIT_MIN=16` | 167.75 / 167.93 (0%) | 208.37 / 207.96 | reject: no gain |
| lop16 `GEPA_ATTN_LOP=16` | — | — | reject at compile: private 400 B, 710 scratch ops (vs 263) |

## dot2 gates (dawn torch fp32 CPU reference)
- Validation modes byte-identical to ctl6: wrap512 fp32kv (1.08e-6) and fp32_bf16kv (1.80e-4), depth8192 fp32kv (strict PASS 1.25e-5,
  0 deviations). Determinism PASS (suite ×2, wrap512 ×3 KV modes, 8192 ×2 fresh). G0 8/8 vs 24f167d.
- **Production bf16-KV changes (relaxed):** wrap512_bf16kv md5 7c6bc13b → bc0423fa. Strict 5.95e-3 → 5.77e-3, forced 2.94e-3 → 3.02e-3,
  routing deviations 35 → 40 (info tier, inside the baseline band).
- ppl: eval_set_full_v1 bf16kv +4.0e-5 → +2.5e-4 (max|Δlogp| 0.261 both). eval_set_v1 identical (32-token prompts never reach the split path).
- Headroom spend on binding tiers: 0. short4/chat vs 2b98737: dot2 0.984 / 0.988, ctl6 0.993 / 0.992 (cross-run noise; rehearsal short +0.7%).

Not self-promoted: the charter's +1.6% both-cases mid rule isn't met (short +0.7%). Projected full @8192 ≈ 34.1 tok/s (>33.42), pending
Alex + an owner-window full-config confirmation.
