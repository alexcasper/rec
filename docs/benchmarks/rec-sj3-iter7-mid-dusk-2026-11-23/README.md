# rec-sj3 iteration 7 — DOT2-base hill-climb, mid-8L on dusk (2026-11-23), source 09192a2

Base d2 = `GEPA_ATTN_SPLIT=1 GEPA_ATTN_LOP=8 GEPA_ATTN_DOT2=1` (mid, SHRINK=1). Lever-off builds (default, split, ctl6, d2, full, full lop8f)
are instruction-identical to the 412c079 controls (0 lines); the diff only adds lines. ABBA 12/12 per row, interleaved order
d2 d2s d2e d2lm d2ns8 d2 d2ns8 d2lm d2e d2s d2. Per-row cotenant gate 22/22 OK with 0 waits; :8080/:8090 up the whole run; max launch 0.196 s.

| build | ctx8192 tok/s | short | vs d2 | scratch ops / private | verdict |
|---|---|---|---|---|---|
| d2 (base) | 173.56 / 173.02 / 172.89 (173.16) | 209.81 / 208.71 / 208.72 (209.08) | — | 263 / 224 B | front-best (Alex-gated) |
| d2s `DOT2_SHORT` (a) | 173.02 / 173.02 | 209.40 / 208.78 | −0.1% / 0.0% | 263 / 224 B | reject: noise; at short depth attention isn't the bottleneck |
| d2e `DOT2_EXP2` (b) | 171.69 / 171.74 | 207.24 / 206.53 | −0.8% / −1.0% | 263 / 224 B | reject: exp count unchanged, extra scale/convert ops |
| d2lm `LASTMERGE` (c) | 152.50 / 151.74 | 182.49 / 181.81 | −12.1% / −12.9% | 274 / 256 B | reject: register spill growth |
| d2ns8 `NSPLIT=8` (c) | 151.23 / 151.29 | 181.57 / 181.50 | −12.6% / −13.2% | 289 / 272 B | reject: register spill growth |

- (c) loses about 13% on the short row too, where the merge/split change isn't on the executed path. The cause is extra spill code across the
  whole hot kernel (+11 / +26 scratch ops), the same class as the iter-5 trap. NSPLIT and LASTMERGE don't combine with DOT2.
- (d) dot4: gfx1150 exposes only `v_dot4_i32_iu8` (int8). `__builtin_amdgcn_dot4_f32_bf8_bf8` needs dot11-insts and fails to compile for
  gfx1150, so there's no float dot4 for bf16 K/V. Probes: `scripts/bisect/dot4probe.hip`, `dot4fp8.hip`.
- Gates not run: no candidate reached the +1.6% acceptance bar. This is a plateau, closed as a negative result.
