# M3 / rec-3cr gated depth-512 row — full Flint, split-K + GEPA_SHRINK=0 (288f4fa default), dusk 2026-11-22

Window: :8080 DOWN 16:43:58Z → run 16:44:03–16:48:02Z (rc 0, longest launch 0.181 s, cap 0.25 s) → :8080 health 200 16:48:28Z;
ops_flint in-band probe after restart: 29.45 tok/s decode @512, prefill 443.09 (~4.7 min total downtime). Stop/done seat-to-seat.
Cotenants per line (window.log): E4B :8090 200, qwen3vl :8990/:8991 200 (loopback), llama-qwen inactive, display inactive,
MemAvailable 16.8–17.0 GiB, VRAM free 38.7 GiB. Lib `flint_full_split_288f4fa.so` sha256 d95d101f3ba4b7be… (instruction stream
identical to the soaked full_split_shr0).

## Verdict: REJECTED (fails the rec-pp2 bf16-KV tier)

| KV build (wrap512 case, 536 pos) | determinism (2 fresh sessions) | vs torch fp32 CPU reference | tier | result |
|---|---|---|---|---|
| fp32kv (mode 2) | PASS | last 8.33e-7, step 1.78e-6, 0 routing devs, tokens exact | 1e-4 | PASS |
| fp32_bf16kv (mode 1) | PASS | last 2.74e-2, step 4.72e-2, 148 routing devs, 1 token beyond near-tie | 1e-3 | **FAIL** |
| bf16kv (mode 0, production; informational) | PASS | last 4.03e-2, step 0.351, 748 routing devs | 1e-3 | FAIL |

Speed rows (measured, NOT quotable as a gated row): ctx512 decode 33.55 tok/s (12/12 stable pairs, depth 512–572; prefill 512 in
14.14 s), short ctx 38.1 tok/s (12/12). Same-box incumbent llama.cpp UD-Q3_K_XL 29.39 @512 (29.45 in-band at restart) → 1.14×,
for reference only.

## Why: not split-K
`kvmode_diff.txt`: kernel fp32-KV and fp32-compute/bf16-KV modes already route differently at prompt position 0 (layer 32), and in
53 of the first 64 positions, where the split branch (GEPA_ATTN_SPLIT_MIN=64) is never taken. fp32-KV passes with split-K active at
1e-6, so the split arithmetic is right. Kernel and reference both round K/V post-norm, post-RoPE with RNE bf16 (`kv_store<bf16>` =
`__float2bfloat16`; torch `.to(bfloat16)`), so there's no obvious convention mismatch. At full config (256 experts, top-6) the bf16-KV
rounding flips routing early and the error compounds. The bf16-KV tier was never run on any full-config build, so the default build
(R11r) is expected to fail it too. That is unverified and needs a window or a decision.

Gate emitter: `scripts/m3_depth_gate.py`; window script: `scripts/m3_gate512_dusk.sh`; scorer `scripts/bisect/score_soak.py`.

## Reference-vs-reference check (offline, dawn): is the 1e-3 bf16-KV tier attainable at full config?
`ref_vs_ref.json` (`scripts/bisect/ref_vs_ref.py`): torch fp32-KV vs torch bf16-KV, both fp32 compute, teacher-forced on the same
wrap512 trajectory, no kernel involved: rel 1.85e-2 at step 0, max 0.146, argmax differs at step 3. The kernel's own fp32-KV vs
bf16-KV modes differ by 1.87e-2 at step 0 and diverge at step 3, the same picture. At full Flint (256 experts, top-6) bf16-KV
rounding is ill-conditioned: implementations that agree to ~1e-6 before rounding hit different bf16 values, flip routing, and
end up about as far apart as the bf16-KV noise itself. So the 1e-3 tier looks unattainable at full config by any
non-bit-exact-torch kernel. That's a gate-spec question for Alex (rec-kcb), not a kernel fix.
