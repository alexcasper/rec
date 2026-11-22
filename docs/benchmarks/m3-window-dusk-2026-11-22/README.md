# Bedivere M3.4: full-Flint head-to-head on dusk (owner window 2026-11-22)

Megakernel: FLINT_CFG_FULL=1, promoted 2b98737 defaults + M3.1 KV continuation + M3.2 split-K attention
(`flint_full.so` sha256 6297889706cc4d54...), bank flint-int4 @3b93f905 (21.02 GiB), bf16-KV production mode, 16 workers.
No override: 139 launches, longest 0.172 s (cap 0.25 s). Incumbent: llama.cpp-Vulkan b4-a37cc46 flint-preview-UD-Q3_K_XL.gguf
on :8080, xh7-r5 procedure (random ids, cache off, 64 greedy, gpu_busy<5 x3 + idle slots).

| row | megakernel decode tok/s | llama.cpp decode tok/s (pre / post) | ratio |
|---|---|---|---|
| depth 512 | **34.40** (12/12 stable ABBA pairs, depth 512..572, pair range 33.0-36.2) | 29.09 / 29.05 / 29.12 ; 28.95 / 28.85 / 28.87 (mean 28.99) | **1.19x** |
| short (depth 4..64) | 39.08 (12/12) | not measured (the shallowest incumbent row is depth 512) | n/a |
| depth 8192 | not run (no ctx8k row staged) | 26.28 / 26.25 ; 26.16 / 26.35 | n/a |
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
