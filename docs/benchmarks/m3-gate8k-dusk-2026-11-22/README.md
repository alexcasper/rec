# rec-kcb window — full Flint, revised full-config tiers, dusk 2026-11-22 18:03–18:28Z

Approved ~25–30 min :8080-down window (stop/done seat-to-seat with ops_flint). :8080 DOWN 18:03:37Z → script 18:03:43–18:28:08Z
(all steps rc 0, longest launch 0.188 s, cap 0.25 s) → health 200 18:28:44Z; ops_flint in-band 29.47 tok/s @512 (~25 min down).
Cotenants per line (`window.log`): E4B 200, qwen3vl 200/200 (loopback), llama-qwen inactive, **display ACTIVE** (it had been
restored since earlier windows; left alone and recorded), MemAvailable 17.1–17.4 GiB, VRAM free 38.7 GiB.
Libs: split `flint_full_split_288f4fa.so` d95d101f… (same .so gated at 512); default `flint_full_def_288f4fa.so` f217f678….
Tiers: `docs/gates/rec-kcb-full-config-tiers-v1.md` (interim, pending Alex).

## A. Default full build @ wrap512: is the bf16-KV divergence a baseline property?  YES
| default build (split off) | vs torch fp32 CPU | |
|---|---|---|
| fp32kv | 6.25e-7 last / 2.14e-6 step, 0 routing devs | PASS |
| fp32_bf16kv | 4.41e-2 / 0.210, 147 routing devs | fails 1e-3, the same as split (2.74e-2, 148) |
| bf16kv | 2.30e-2 / 0.271, 756 routing devs | same picture |
Default runs 41 ms/position at 512 (split 30).

## B. ppl tier, split+SHRINK=0, eval_set_full_v1 (8 × 192 real text)  PASS
| mode | kernel ppl | ref ppl | rel | limit |
|---|---|---|---|---|
| fp32kv | 3.28033 | 3.28033 | +1.5e-7 | 1e-4 PASS |
| bf16kv (production) | 3.2871 | 3.2760 (bf16-KV ref) | +3.4e-3 | 1e-2 PASS |
Reference bf16-KV vs fp32-KV ppl: −1.3e-3.

## D/C. split+SHRINK=0 gated depth-8192 row  REJECTED (fails the binding fp32-KV tier)
| check | result |
|---|---|
| fp32kv @8192 vs torch fp32 CPU | **FAIL** last 7.41e-2, step 0.297, 3 routing devs, tokens exact (degenerate repeating trajectory) |
| determinism, production bf16kv @8192 | PASS (2 fresh sessions byte-identical, md5 6d6ae448…) |
| determinism, fp32kv @8192 | not tested (D2 skipped by the 27 min hard stop) |
| speed, production @8192 | 21.55 tok/s (12/12 stable pairs, depth 8215–8251, k=2) vs incumbent 26.80 = **0.80×** (display active) |
| bf16-KV informational @8192 | not scored (scoring stopped to free memory for the binding question) |

### Diagnosis so far
- **Not RoPE convention:** with the reference's RoPE inv_freq replaced by the kernel's exact exp2f formula, the error is unchanged (7.415e-2;
  `D_8k_rope_mimic.json`), although the formulas differ by 4.9e-4 rad at depth 8215.
- **Not a partial-buffer overflow:** n_split is capped at 16, so partials take 32 × 16 × 130 floats at any depth, against a 150144-float buffer.
- **Mid at depth 8192** (no window, `mid8k_*_scores.json`): default PASS 8.1e-6 / 1.2e-5, 0 routing devs; split 2.1e-5 / 3.7e-5 with
  **1 routing deviation** → FAIL. The reference is sound at depth. Split-K's relaxed merge carries ~2.6× the default's depth error, enough to
  flip a route at mid. At full config (48 layers, 256 experts top-6) a few flips compound, which fits the 7.4e-2. Telling split-amplified
  error apart from a baseline full-depth error needs the default full build fp32kv @8192 in another window.

### What the row means for rec-e5d
Correctness aside, the speed is an upper bound: split-K at depth 8192 runs 21.55 tok/s against llama.cpp's 26.80 (0.80×), the same as the
earlier 8k slot (21.35). The split-K depth-attention lever, as built (n_split ≤ 16, one warp per (head, chunk)), does not close the depth
gap. rec-e5d needs a different design, not this row.

## Tooling notes
- The gate tool now sets the driver's `_CARD`. In the gate512 window the busy gate and VRAM preflight were silently disabled (VRAM was
  fine, 38.7 GiB free, logged independently by the window script).
- In-session decode replay is NOT a determinism check: the 513-slot sliding-window KV ring is overwritten by the first decode.
