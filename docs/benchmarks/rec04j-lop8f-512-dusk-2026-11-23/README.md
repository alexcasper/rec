# rec-04j — @512 decode re-baseline on lop8f (SHRINK=1+fence, single build for both depths), dusk, 2026-11-23

Lib `~/rec-ve/fut11b/lop8f/flint_mid.so`, sha256 ec34a71153bae74b… (rec-04i build at ee04ff3: `FLINT_CFG_FULL=1 GEPA_ATTN_SPLIT=1
GEPA_ATTN_LOP=8`, SHRINK=1 + rec-11b VMEM fence; VGPR 256, scratch 288 B). Script `scripts/rec04j_window_dusk.sh` (self-waiting,
lib-sha check, VRAM/busy/llama-qwen preflight, per-line cotenant log). Protocol identical to R11r / gate512: `flint_full_driver.py --window-bench
--rows short,ctx512 --pairs 12 --mode 0` (production bf16-KV). det512 already PASS (stageC); not repeated.

## Window
- Pre-check 14:37:32Z: busy 0/0/0, MemAvailable 16.4 GiB, 4 llama-servers, :8080 200. Script armed 14:37:42Z; STOP sent seat-to-seat to ops_flint (w11:p1Q).
- **PAUSED 14:37:57Z** (ops_flint) → script saw the window open at 14:38:02Z (VRAM free 38.7 GiB) → bench 14:38:05–14:39:31Z, rc 0, longest launch 0.162 s.
- **RESTARTED ~14:40:08Z** (ops_flint: total downtime **2m11s**, in-band 29.53 tok/s decode / 454.49 prefill). Health 200 from dawn at 14:40:49Z.
  On-box probe 14:41:04Z: 30.23 / 30.32 tok/s.
- Cotenants per line in `window.log`: E4B 200, qwen3vl 200/200, llama-qwen inactive, display active.

## Rows (12/12 stable ABBA pairs each)
| row | lop8f | R11r default (5363ad5, stale) | gate512 split+SHRINK=0 (288f4fa) | incumbent llama.cpp UD-Q3_K_XL |
|---|---|---|---|---|
| ctx512 decode (depth 512..572) | **37.69 tok/s** | 24.75 | 33.55 | 29.39 |
| short decode (depth 4..76) | 38.75 | 39.06 | 38.1 | — |
**ctx512: 1.282× vs incumbent 29.39** (R11r was 0.84×). +52% vs R11r, +12.3% vs the gate512 split build. Prefill 512 in 13.39 s.
With rec-04i's @8192 row (33.42 tok/s), lop8f is one build that wins at both depths: 1.28× @512, 1.25× @8192 vs incumbent.
Promotion remains Alex-gated (rec-kcb). The coordinator is scoring rec-04i's fp32kv emit (not duplicated here).
