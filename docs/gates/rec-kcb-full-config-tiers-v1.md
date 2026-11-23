# rec-kcb ruling — full-config (FLINT_CFG_FULL) acceptance tiers, v1

**Status: RATIFIED by Alex via Bedivere 2026-11-23 (rec-kcb CLOSED): KEEP THE MEGAKERNEL LINEAGE —
bf16-KV is REJECTED for the full 18.5 GB config (fails rec-pp2 1e-3 tier at 2.74e-2 from position 0,
rounding-boundary amplification per eb06650, unattainable by any non-bit-exact kernel); fp32-KV is the
run-of-record. The open question is resolved: full-config binding tiers = fp32-KV max-rel (rec-pp2 1e-4
tier) + perplexity delta vs the stock reference — now codified in
`engine/validation/megakernel_mid/qgate/gate_spec_full_v1.json`, which serve/eval pipelines check
explicitly (production row = fp32-KV). No further bf16-KV attempts at full config unless Alex reopens.**

## Evidence
`docs/benchmarks/m3-gate512-dusk-2026-11-22/ref_vs_ref.json`: on the same full-Flint wrap512 trajectory, torch fp32-KV vs torch
bf16-KV (both fp32 compute, no kernel) differ by 1.85e-2 at step 0, max 0.146, with an argmax flip at step 3. That is about 18× the
1e-3 bf16-KV logit tier. The kernel's own fp32-KV vs bf16-KV modes show the same gap (1.87e-2, divergence at step 3), with
routing differences from prompt position 0. At 256 experts / top-6, bf16-KV rounding is ill-conditioned: correct implementations
that agree to ~1e-6 before rounding diverge to noise scale. No non-bit-exact kernel can attain the tier, so it can't separate
good kernels from bad ones at full config.

## Binding tiers at full config
1. **fp32-KV logits:** kernel mode 2 (fp32 compute, fp32 KV) vs torch fp32 CPU reference, max-rel <= 1e-4 on last and step
   logits, with routing judged through the kernel's step_experts (mid_mk_check spec (a)). Cases: wrap512 plus every depth case a row is quoted at.
2. **N-repeat determinism:** >= 2 independent emits (fresh sessions, full bank re-stream) byte-identical, for every KV mode
   (fp32kv, fp32_bf16kv, bf16kv) on every gated case.
3. **Perplexity:** teacher-forced ppl on the frozen full-config set `qgate/eval_set_full_v1.json`
   (8 real-text items × 192 tokens, Flint tokenizer, sha256 1b06d928…): production bf16-KV mode |ppl_rel| < 1% vs the torch
   fp32-compute reference with bf16 KV store; fp32-KV mode |ppl_rel| <= 1e-4 vs the fp32 reference.
   (eval_set_v1's ppl items are samples from the synthetic mid checkpoint, so they're not meaningful for real Flint. 192 tokens put
   2/3 of positions on the split-K branch.) Emitter: `scripts/m3_depth_gate.py --ppl-set`; scorer: `qgate/score_ppl_full.py`
   (protocol validated on mid: fp32kv rel 1.8e-8).

## Rejected at full config (Alex ruling 2026-11-23)
- **bf16-KV 1e-3 logit tier: REJECTED — the bf16-KV production mode is dead at full config.** Evidence:
  2.74e-2 vs 1e-3 from position 0 (148 routing devs on wrap512); eb06650 root cause (rounding-boundary
  amplification; torch ref-vs-ref gap 1.85e-2, argmax flip at step 3) makes the tier unattainable by any
  non-bit-exact kernel. Informational bf16-KV readings may still be recorded per row, never binding.
  Candidate note: the DOT2 lever compiles only into the bf16-KV path, so it is a no-op for the shipped
  fp32-KV config (rec-dor amendment).

## Unchanged
- Mid config: rec-pp2 two-tier (fp32-KV 1e-4 / bf16-KV 1e-3) + ppl < 1% on eval_set_v1, as before.
- rec-1f4 qgate T1–T4 stays mandatory pre-promotion evidence (mid).
- Full-config SHRINK state: the SHRINK=0 pin was replaced by the explicit VMEM boundary fence in rec-11b
  (98ce41d/ee04ff3, SHRINK=1 deterministic at full config, rec-04i row 33.42 @8192 / 37.69 @512 in
  production bf16-KV mode). `-DGEPA_SHRINK=0` remains the recipe ONLY for instruction-identical
  reproduction of the measured 8816eebf lib (rec-04i pin).
