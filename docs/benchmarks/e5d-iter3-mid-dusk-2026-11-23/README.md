# rec-e5d hill-climb iteration 3 — mutations of lop4, mid-8L ctx8192 on dusk (2026-11-23)

Source: generic lanes-over-positions block (`GEPA_ATTN_LOP` ∈ {4,8}, `GEPA_ATTN_LOP_PF` register prefetch of the next iteration's K/V,
`GEPA_ATTN_LOP_UNROLL`). The default, split, full, full+split and gqa builds have the same instruction streams as the previous HEAD. All builds use
`GEPA_SHRINK=0 GEPA_ATTN_SPLIT=1`. A = the measured lop4 lib from 390c4cd. lop4r = the same knob compiled from the generic block (equivalence control).
:8080 up. Cotenants idle on every pair (row 1 started 02:35:26Z, 41/41 samples flint idle, after ops_flint's restart probe).

| build | VGPR | ctx8192 tok/s (12/12 ABBA each) | mean | vs lop4 | short |
|---|---|---|---|---|---|
| lop4 (A) | 245* | 154.73 / 154.51 / 154.69 / 154.47 | 154.60 | — | ~207 |
| **lop8** (8 positions/iter, 4 lanes × 32 dims) | 254 | 166.87 / 166.77 | **166.82** | **+7.9%** | ~207 |
| lop4pf | 256 | 158.96 / 158.57 | 158.77 | +2.7% | ~208 |
| lop4u2 | 248 | 154.32 / 154.25 | 154.29 | −0.2% | ~208 |
| lop4r (control) | 248 | 154.00 / 154.71 | 154.36 | −0.2% | ~208 |
*full-config VGPR. Run order: lop4 lop8 lop4pf lop4 lop4u2 lop4r lop4 lop8 lop4pf lop4u2 lop4r lop4. Max launch 0.195 s.

## lop8 gates (mid; dawn torch fp32 CPU reference)
- Determinism PASS: suite ×2, wrap512 ×3 KV modes, 8192 fp32kv ×2 fresh sessions. G0 8/8 vs 24f167d.
- wrap512: fp32kv 1.08e-6 PASS; fp32_bf16kv 1.80e-4 PASS (mid-binding 1e-3); prod bf16kv 5.95e-3 strict FAIL (baseline property: def0 5.91e-3).
- **8192 fp32kv: strict PASS 1.25e-5, 0 routing deviations.**
- ppl: v1 −3.5e-8 / −6.9e-4; full +1.4e-8 / +4.0e-5. PASS.
- short4/chat vs 2b98737: 0.9776 / 0.9811. That fails strict non-regression, the same as every SHRINK=0 build (def0 0.9747 / 0.9807);
  it's better than lop4 (0.970 / 0.971).

## Projection to full @8192 (unmeasured)
Per full-attention-layer overhead: lop8 (1/166.82 − 1/207) s ÷ 4 = 0.29 ms (lop4 0.41 ms measured-consistent). Projected full
≈ 1 / (26.2 ms + 14 × 0.29 ms) ≈ **33.0 tok/s (≈1.23× vs 26.80)**. lop4's projection of 31.4 measured 31.36. This needs a full-config
window to confirm.

## Next
lop8 is the iteration-3 front-best on mid. lop8 + PF would need more registers, and lop4pf already sits at 256 VGPR. A full-config lop8 row needs
an owner window. Promotion is Alex-gated (rec-kcb).
