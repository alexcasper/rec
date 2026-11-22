# Bedivere milestone 3: full-Flint megakernel on dusk vs the llama.cpp-Vulkan incumbent (feasibility + plan)

Status: PLAN (rec-ohs pivot, 2026-11-22). Nothing on dusk touched for this yet. Written after Run 4: the dusk mid-8L
lane's last residual lever landed (c033 hot-code shrink, promoted as 2b98737: 214.3 tok/s, byte-identical). The
coordinator ranks the full-Flint decision benchmark above further ~3 % dusk micro-gains. Parked follow-up: more
hot-code shrink (the I-cache axis is now confirmed: -974 instructions = +2.3-3.6 %).

## What already exists (do not redo)

- **Full-config kernel**: `FLINT_CFG_FULL=1` build of the same megakernel (49 L, 256 experts, top-6, dense [0,1],
  14 full-attention layers). It was validated GREEN on albion R9700 (rec-u9q, e3c5ed6): decode matrix R8 =
  **35.45 tok/s decode32 after 512** (w64), short-ctx 54.1/67.6/69.9.
- **Bank**: `flint-int4.bank` (21.02 GiB, bit-exact, from RecursiveSI/flint-int4@3b93f905) plus
  `flint-int4.off.npy` are **already on dusk** in `~/xh7/`. The offsets are int64, so >4 GiB indexing is fine.
- **Incumbent on dusk**: llama.cpp-Vulkan serving `flint-preview-UD-Q3_K_XL.gguf` (18.49 GB) on :8080 = R5,
  **26.61 tok/s decode at 8k depth** (29.12 at depth 512). The "31.1 @ 8k" figure in the dispatch was RETIRED in
  the xh7 reconcile: its "8k" was the context window, not the depth. 26.61 is the only current comparable.

## Ruling carried into M3 (flag)

The megakernel consumes **uniform INT4-affine g64** (flint-int4) only: Alex's standing rule (INT4-affine only;
Q3_K dequant rejected) and the rec-are layout ruling. So the head-to-head is **megakernel @ flint-int4 (21.0 GiB,
~4.5 bpw)** vs **llama.cpp @ UD-Q3_K_XL as shipped (18.5 GB, ~3.5-3.9 bpw)**. That is the real decision
benchmark, and the incumbent moves fewer bytes per token. Every row carries a bytes/token column. The
byte-matched llama.cpp Q4_0 row (R10 analogue) is optional context.

## Feasibility read

| question | finding | consequence |
|---|---|---|
| weights layout | same s2 layout as mid-8L; every phase shape is validated on mid-8L (hid 2048, K in {512,1024,2048,4096,6144}) | no kernel layout work |
| DRAM footprint | dusk has 48 GiB VRAM carve-out, **33.2 GiB in use** (flint :8080 + E4B + two qwen3vl servers); 14.8 GiB free; GTT 19.5 GiB free; system RAM 6 GiB available | the 21.0 GiB bank **cannot be resident next to the current cotenants**. M3 windows need flint :8080 stopped (it frees ~18.5 GB): an **ops_flint owner window** |
| KV at 8k | bf16 KV: 14 full layers x 2 x 4 x 128 x 8192 x 2 B = 235 MB, plus 35 sliding x 1.05 MB = 37 MB | fits easily |
| kernel arg plumbing | `mid_session_run_ex2` restarts at position 0 every launch (`for pos = 0..total`); the session owns KV buffers but cannot continue | **needs start_pos + KV continuation** to build an 8k context in launches of <= 0.25 s (the dusk display-iGPU wedge rule, rec-u9q) |
| prefill to 8k | the megakernel prefill is token-serial at decode speed; ~8k positions at ~25-40 ms = ~4-5 min in ~1000 chunked launches | acceptable for a benchmark once continuation exists |
| multi-buffer | one 21.0 GiB hipMalloc in VRAM is fine once :8080's memory is released; no split needed. If VRAM stays fragmented, fallback = per-layer buffers + a pointer table in place of `bank + off[i]` (Q_W/Q_S/Q_B are the only consumers) | contingency only |
| driver naming | `flint_full_driver` expects `<bank>.off.npy`; the file on disk is `flint-int4.off.npy` | one-line fix (rec-are D) |
| correctness host | the full-config torch fp32 CPU reference ran on albion. dawn: 46 GiB (25 available), dusk: 6 GiB available | full-config rec-pp2 stays **albion-gated**, or a streamed per-layer CPU reference on dawn (to verify) |

## Projection (from the measured dusk profile, not an estimate of the kernel)

Bytes/token, full Flint INT4: 47 MoE layers x 25.8 MB (QKV 5.9 + O 4.7 + router 1.0 + gate/up 9.4 + down 4.7)
+ 2 dense x 31.9 MB + lm_head 173 MB = **~1.45 GB**, plus ~0.27 GB KV reads at 8k depth = **~1.72 GB/token**.
At the 82 GB/s roof that is a 47.6 tok/s ceiling, ~40 tok/s at the measured 84 % weight-phase efficiency. The
incumbent is 26.61.

**Blocker in that projection: attention at depth.** The attention phase is one warp per q-head, serial over
positions (32 of 128+ warps busy). R8's own drop from 69.9 (short) to 35.45 (after 512) on R9700 is that cost.
At 8k depth it would dominate (rough: tens of ms/token across 14 full layers). So **split-K (flash-decoding)
attention across all warps is a prerequisite**, not an option, for an 8k-depth head-to-head. It is relaxed
(softmax-merge order). Gate it on mid-8L first: rec-pp2 two-tier + ppl, one rationed relaxed change against the
6-7e-5 fp32_bf16kv margin.

## Plan (each step gated; nothing on dusk before step 1 is reviewed)

1. **M3.1 plumbing (desk + dusk mid-8L only)**: `start_pos` / KV continuation in `mid_session_run_ex2` (byte-identity
   check: a 36-token run as 1 launch vs 4 continued launches must give identical npz on mid-8L); the off-table naming fix.
2. **M3.2 attention split-K** on mid-8L: depth sweep (short / 512 / 2k with chunked launches) + rec-pp2 + ppl.
   Accept only if correct and if it flattens the depth curve.
3. **M3.3 full-config correctness**: albion when it is back (rec-pp2 full matrix), or a streamed dawn CPU reference
   if feasible. No full-config number is reported as a result before this passes.
4. **M3.4 owner window (ops_flint)**: stop :8080 for the window, load the bank, chunk-prefill to 8k, decode32 at
   8k depth (ABBA-style interleave with the incumbent's measurement taken in the SAME cotenant state either side of
   the window), then restart :8080 per the owner. The row goes into the xh7 decision matrix with bytes/token.

## Open questions for the coordinator / Alex

- Confirm the comparison basis: INT4-affine megakernel vs UD-Q3_K_XL incumbent (different bytes) is the decision
  benchmark; or also run a byte-matched llama.cpp Q4_0 on dusk.
- ops_flint window for :8080 (memory), and whether E4B/qwen3vl must also be down (VRAM headroom after :8080 stops:
  ~33 GiB free, enough for 21 GiB bank + KV + workspace).
- Correctness host for full config while albion is offline.
