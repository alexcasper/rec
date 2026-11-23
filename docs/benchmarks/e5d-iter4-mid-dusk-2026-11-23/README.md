# rec-e5d hill-climb iteration 4 — mutations of lop8, mid-8L ctx8192 on dusk (2026-11-23)

Source 4ba4c94 (no kernel change this iteration). Builds `GEPA_SHRINK=0 GEPA_ATTN_SPLIT=1 GEPA_ATTN_LOP=8` plus:
lop8pf `GEPA_ATTN_LOP_PF=1` (**spills**: VGPR 256, private 240 B, 308 scratch ops vs 6), lop8u2 `GEPA_ATTN_LOP_UNROLL=2` (VGPR 254, no spill),
lop8c = the iter-3 lop8 .so re-run as control (md5 78ba73e295). :8080 up. Cotenants idle on every pair (flint and e4b idle 287/287 samples).

| build | ctx8192 tok/s (12/12 ABBA each) | mean | vs lop8 | short |
|---|---|---|---|---|
| lop8 (control) | 167.14 / 167.53 / 167.11 | 167.26 | — | 207.7 / 207.05 / 206.73 |
| lop8u2 | 167.18 / 167.55 | 167.37 | +0.07% (noise) | 207.5 / 207.38 |
| lop8pf | 166.21 / 166.33 | 166.27 | −0.6% | **201.77 / 201.19 (−2.8%, spill)** |
Run order: lop8 lop8pf lop8u2 lop8 lop8u2 lop8pf lop8. Max launch 0.196 s. The iter-3 lop8 reading was 166.82; this iteration's control reads 167.26.

## Gates (rec-pp2 + mid strict 8192)
**Every gate emit of lop8pf, lop8u2 and lop8c is byte-identical to iter-3 lop8**: wrap512 fp32kv abf6523c85 / fp32_bf16kv 8fe4824b2b /
bf16kv efa072c3ba, depth8192 fp32kv 14814c8311, ppl v1+full logp in both KV modes. Prefetch and unroll change load scheduling, not arithmetic
order, so all gate results equal lop8's (iter 3):
- determinism PASS (suite ×2, wrap512 ×3 KV, 8192 fp32kv ×2 fresh); G0 8/8 vs 24f167d.
- wrap512 fp32kv 1.08e-6 PASS; fp32_bf16kv 1.80e-4 PASS; prod bf16kv 5.95e-3 (baseline property).
- **mid strict 8192 fp32kv PASS 1.25e-5, 0 routing deviations.**
- ppl v1 −3.5e-8 / −6.9e-4, full +1.4e-8 / +4.0e-5 PASS (npz in `ppl/`; reproduced via rescore_ppl.py).
- short4/chat vs 2b98737: lop8pf 0.9729 / 0.9701, lop8u2 0.9734 / 0.9735 (SHRINK=0 pin class; def0 0.975/0.981).

## Verdict
No improvement in iteration 4. Neither mutation clears the 1.6% bar, and lop8pf regresses short decode by 2.8% through register spill (a
scratch-traffic build is also undesirable under rec-3cr). **lop8 stays front-best.** The register-level knobs on the lanes-over-positions
loop are exhausted: unrolling the variable-trip loop changes nothing, and prefetch costs more in spill than it saves in latency.
