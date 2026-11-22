# Bedivere M3.2: split-K (flash-decoding) attention, dusk mid-8L (2026-11-22)

Knob `GEPA_ATTN_SPLIT=1` (default 0), `GEPA_ATTN_SPLIT_MIN` (default 64). When the context exceeds the threshold, (head, chunk)
warps (chunks = min(16, groups/heads)) write partials (acc, max, denom) into the free lm_head logits buffer; one extra
grid barrier; a merge folds in the sink. Relaxed form: only the softmax merge order changes.

| build | short4 / chat vs promoted 2b98737 (ABBA x8) | ctx512 decode (continuation, 8 pairs, def/B/B/def) | hot instr |
|---|---|---|---|
| default (M3.1) | (1.001 / 0.990) | 134.03, 134.17 tok/s | 10959 |
| **splitB (MIN=64), candidate** | 0.991 / 0.980 (byte-identical outputs; code size) | **180.53, 179.26 tok/s (+34%)** | 11842 |
| splitA (MIN=0, forced) | 1.005 / 0.977 | n/a | 11842 |

Gates (torch fp32 CPU reference on dawn, mid_mk_check):
- splitB: suite + ppl npz 8/8 byte-identical to 24f167d, so short4/chat and ppl are inherited. Binding margin fp32_bf16kv short4
  stays 9.115e-4; headroom ration spend = 0. wrap512 (512 + 24, continuation emit, longest launch 0.239 s): fp32kv 1.40e-6
  (def 1.56e-6), fp32_bf16kv 1.935e-4 (def 1.816e-4; +1.2e-5, far under 1e-3), bf16kv 6.68e-3 (def 5.91e-3; both fail the
  absolute bf16 tier, the known T2 baseline issue; ratio 1.13 < 1.5 non-regression). Tokens identical to default at all three builds.
- splitA (forced at every length): short4 fp32_bf16kv 9.739e-4 would spend 6.2e-5 of the ~6-7e-5 ration, so it is NOT the
  promoted form. ppl fp32kv +1.1e-8, bf16kv -1.04e-3 relative (< 1%).
Use: the M3 full-config window build (`flint_full.so`) is built with `-DGEPA_ATTN_SPLIT=1`. Mid-8L production default stays 0
(no short-context gain, -1 to -2% from code size).
