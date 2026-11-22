# Bedivere M3.4: full-Flint head-to-head on dusk (owner window 2026-11-22)

Megakernel: FLINT_CFG_FULL=1, promoted 2b98737 defaults + M3.1 KV continuation + M3.2 split-K attention
(`flint_full.so` sha256 6297889706cc4d54...), bank flint-int4 @3b93f905 (21.02 GiB), bf16-KV production mode, 16 workers.
No override: 139 launches, longest 0.172 s (cap 0.25 s). Incumbent: llama.cpp-Vulkan b4-a37cc46 flint-preview-UD-Q3_K_XL.gguf
on :8080, xh7-r5 procedure (random ids, cache off, 64 greedy, gpu_busy<5 x3 + idle slots).

| row | megakernel decode tok/s | llama.cpp decode tok/s (pre / post) | ratio |
|---|---|---|---|
| depth 512 | **34.40** (12/12 stable ABBA pairs, depth 512..572, pair range 33.0-36.2) | 29.09 / 29.05 / 29.12 ; 28.95 / 28.85 / 28.87 (mean 28.99) | **1.19x** |
| short (depth 4..64) | 39.08 (12/12) | not measured (the shallowest incumbent row is depth 512) | n/a |
| depth 8192 (8k slot 13:39-13:46Z) | **21.35** (12/12 stable pairs, depth 8192..8228, k=2) | 26.17 / 26.28 ; 26.38 / 26.37 (mean 26.30; 12:22 window 26.28 / 26.25 ; 26.16 / 26.35) | **0.81x** |
| prefill | 37 pos/s (serial decode-path prefill) | 448-456 (512), 441-443 (8192) | 0.08x |
| weight footprint | 21.02 GiB (22.57 GB) INT4-affine g64 | 18.49 GB UD-Q3_K_XL | 1.22x bytes |

Cotenant state:
- megakernel rows: `flint:down,e4b:idle,qwen:down` on every pair (llama-qwen = the disabled :8091 unit), qwen3vl 8990/8991 up idle,
  MemAvailable 8.5-8.7 GiB, host staging 256 MiB (bank streamed 9.1 s at 2.30 GiB/s).
- incumbent rows: 8080 up, 8090 up idle, 8990 up, llama-qwen inactive, busy 3 before each row.
- matched except :8080 itself.

Timeline (UTC):
- 11:29 30-min signal sent; the first pre-stop row ran at 9.99 tok/s under a foreign harvest_dusk.py qwen3vl GPU load. VOIDED
  (`incumbent_rows.VOID-1130Z-harvest.jsonl`).
- 12:21 harvest ended; 12:21-12:23 pre-stop rows.
- 12:25:06 ops_flint stopped :8080 (VRAM free 37.7 GiB).
- 12:25:35 m3_window_dusk.sh: bank streamed, spot written. The bench then refused on its own VRAM preflight: free memory
  read 16.35 GiB because the spot session's 21 GiB hadn't shown as freed in sysfs before the second session's check.
  Fix follow-up: one session for spot + bench.
- 12:26:30 bench rows re-run alone in a fresh process (identical args minus --emit-spot): `driver_bench.log`, `window_bench.json`.
- 12:28:29 DONE; :8080 restarted ~12:29 (ops_flint in-band 29.24 decode / 441.95 prefill); 12:29-12:31 post-restart rows.

Caveats:
- Small k (B = {last}+4 tokens, sized to the 0.2 s target at ~29 ms/position), so the per-pair spread is about ±5%; 12 pairs.
- The spot fp32kv short4 npz (`spot/` on dusk) is NOT yet scored against a CPU reference. That is M3 step 3 (streamed CPU
  reference on dawn), so the megakernel rows are perf-only until it passes.
- No 8k megakernel row was staged.

## Preflight refusal: root cause (added after the coordinator's flag)

Files: `driver.log` is the staged script's first process: spot + REFUSED at 12:25:49. `driver_bench.log` is the second process
(12:26:33-12:28:29), which ran BOTH rows: ctx512 34.40 (12/12) and short 39.08 (12/12). Its last line is `window bench ok`.
`window.log` stops at its `start:` line: the script's `rc=${PIPESTATUS[0]}` never ran under `set -e` (fixed).
Cause: NOT double-counting and NOT a carve-out shortage. emit_spot frees its session (hipFree of the 21 GiB bank) and the
bench's preflight ran in the same second (12:25:49). sysfs `mem_info_vram_used` releases freed buffers asynchronously.
Reproduced on dusk with the mid bank: free 15.69 GiB right after mid_session_free, 16.71 GiB 2 s later. So 16.35 = 37.53 - 21.18
was the spot bank still counted. The ":8091 stop" advisory was stale text (8091 down throughout).
Fix: the preflight re-reads sysfs for up to 15 s before refusing; the advisory names the cotenant state instead; the script
records the driver rc.
Still missing on the megakernel side: depth 8192 (never staged). The serial decode-path prefill of 8192 positions at
~30-40 ms/position is ~4-6 min, so an 8k row needs a dedicated window slot (or a batched prefill path) to fit the
~8 min / 10 min SLA.

## Depth-8192 slot (coordinator call, 2026-11-22 13:39-13:47Z)

`scripts/m3_window8k_dusk.sh` with the fixed driver (4073d7d + ctx8192 row). No override; 1 process; preflight passed first
read (need 21.86 GiB, 37.53 free). Bank streamed 9.0 s; serial prefill 8192 positions in 305.5 s (deadline 420 s): 28.7 ms/position
at depth 0, 47.8 ms at depth 8192. 12/12 stable pairs, `flint:down,e4b:idle,qwen:down` on every pair, MemAvailable 8.3-8.4 GiB.
Max launch 0.170 s; span 367 s. Pairs (tok/s): [23.54, 20.68, 21.93, 20.48, 22.53, 20.21, 21.85, 19.99, 23.36, 19.83, 22.4, 20.26]. AB ~22-23 vs BA ~20 is the order effect that ABBA averages out;
k=2 (1-token difference) because B must stay <= 0.2 s at 47.8 ms/position.
Timeline: pre-stop rows 13:38; :8080 DOWN 13:39:35Z (ops_flint); launch 13:39:38Z; DONE 13:45:45Z; health 200 13:46:28Z (dawn watcher);
ops_flint in-band 26.32 decode / 442.95 prefill at depth 8192; post-restart rows 13:46-13:47. :8080 downtime 6m53s.

Reading: decode at 8192 costs 46.8 ms/token vs 29.1 at 512 (+17.7 ms). The 14 full-attention layers read
14 x 8192 x 4 kv-heads x 128 x 2 x 2 B ~ 235 MB/token of bf16 KV, ~3.4 ms at 70 GB/s. So the depth attention runs ~5x off the
bandwidth roof: the split-K path is latency-bound (one position per warp-iteration with a 32-lane reduction, splits capped at
min(16, groups/heads)). This is the lever for the 8k row, not the weights.

## Window 3: display session stopped (Alex option b), workers sweep (2026-11-22 13:59-14:10Z)

ops_flint stopped :8080 at 13:59:31Z and the display session (DM + Plasma/kwin/Xwayland/chrome). MemAvailable 8.9 -> 17.0 GiB,
VRAM free 38.7-39.7 GiB, GPU busy 0. Display stays down through the weekend (restore by Tue 2026-11-24, ops_flint backstop).
`scripts/m3_sweep_dusk.sh` launched 14:06:37Z, once ops_flint's DOWN confirmation landed; done 14:09:40Z (rc 0, span 183 s, no override,
max launch 0.173 s). Cotenants on every pair: `flint:down,e4b:idle,qwen:down`, kwin=0; MemAvailable 17.0-17.2 GiB (`sweep/window.log`).

| workers | ctx512 decode tok/s | short decode tok/s | pairs |
|---|---|---|---|
| 16 | **34.74** | **39.35** | 12/12 + 12/12 |
| 12 | 32.49 | - | 8/8 |
| 8 | 28.78 | - | 8/8 |

16 workers (the co-resident maximum) is best. Display-down vs display-up: 34.74 vs 34.40 @512, 39.35 vs 39.08 short (+1%).
Incumbent post-restart, display down (14:10-14:12Z): depth 512 29.25 / 29.46 / 29.45 (mean 29.39), depth 8192 26.82 / 26.77 (mean 26.80);
ops_flint in-band 29.35 @512. No display-down pre-stop rows: the earlier pre-stop rows were display-up.
Display-down pair lines: @512 34.74 vs 29.39 = **1.18x**; @8192 megakernel not re-measured with the display down (the 5 min prefill
didn't fit beside the sweep), so the display-up 21.35 stands against 26.80 display-down / 26.30 display-up (R5 record 26.61) = **0.80-0.81x**.
:8080 downtime 13:59:31 -> 14:10:18Z (health 200, dawn watcher) = 10m47s. About 7 min of that was before launch: ops_flint's
DOWN confirmation (display teardown + backstop cron) reached this seat at 14:06; the bench itself was 3m03s.

## R11 INVALIDATED; R11r re-run on the verified build (rec-3cr, 2026-11-22 15:09-15:11Z)

Every megakernel row above ran on `flint_full.so` (HEAD + GEPA_ATTN_SPLIT=1), which rec-6ek showed is a nondeterministic,
incorrect build at full config. Those rows are unquotable. Re-run on `full_k_5363ad5.so` (HEAD default, sha 3ab8a281...):
fp32-KV PASS (1.55e-6), 5/5 repeats bit-identical in this window (8/8 total). w16, display down, :8080 down 15:09:04-15:11:17Z.

| row | megakernel (verified) | llama.cpp (display down) | ratio |
|---|---|---|---|
| decode32 @512 | **24.75** (12/12, depth 512-548) | 29.39 | **0.84x** |
| short (depth 4-76) | 39.06 (12/12) | n/a | - |
| @8192 | no valid row | 26.80 | - |

The "1.18x @512" was entirely the racy split-K path. w32 is not possible on dusk (max co-resident workers = 16).
