# rec — AMD ROCm megakernel fork (Recursive Flint on gfx1150)

This fork ports the `rec` megakernel to **AMD Radeon GPUs** — development
target: Radeon 890M (gfx1150, Strix Point) on the fleet's `dusk` box, running
Recursive Flint mid (8-layer) and full (49-layer, 18.5 GB) configurations.
Upstream (`recursive-oss/rec`) remains the Apple Silicon lineage; this branch
(`rec/gemini-kernels`) carries the AMD HIP megakernel, the correctness-gate
apparatus, and the quantization studies. Branch: `rec/gemini-kernels` ·
bench host: dusk · co-tenant discipline with ops_flint serving.

## Accomplished (all numbers verified; artifacts in `docs/benchmarks/`)

### Decode throughput records

| config | variant | tok/s @8192 | vs reference |
|---|---|---|---|
| full Flint 49L | lop8 fp32-KV, SHRINK=1+fence (**rec-04i**, run-of-record) | **33.40** | **1.25×** llama.cpp-Vulkan (26.80) same-box; 1.07× solo baseline (31.1) |
| mid 8L | int8-KV v1.1 (**rec-ywy v1.1**) | **162.26** | +7.4% vs fp32kv (151.14), +2.5% vs int8-KV v1 (158.27) |
| mid 8L | megakernel q4 (original run-of-record) | 120.78 | 32.7× eager-torch INT4 (3.69) |
| mid 8L | short-context decode | ~205 | — |

Determinism: promoted variants pass 3×-fresh-session identical-output specs
(rec-04i: clean; int8-KV v1.1: det ×2 PASS, md5-matched).

### int8-KV cache (mode 3)

fp32 compute with per-token-head symmetric int8 KV (scale = absmax/127 at
write, RNE, dequant on read). Read paths implemented: 1-warp (sinks), split
non-LOP, split LOP. **GQA, LDS, LOP_PF compile-time rejected — that is the
full-config work (below).** v1.1 folds row scales out of the LOP inner loop.
Kernel-vs-torch agreement 99.6% top-1 at matched quantizer. Determinism recipe
pinned: build int8-KV with `-DGEPA_SHRINK=0` (SHRINK=1 + int8 nondeterminism
root-caused open as `rec-0v1`; fence v2 + SHRINK=0 both det-pass at mid).

### KV-precision ladder study (rec-hyf)

Teacher-forced quality vs fp32-KV on the held-out eval set:

| tier | top-1 | ppl Δ | verdict |
|---|---|---|---|
| bf16 (2 B) | 98.6% | −0.13% | best 2× memory tier |
| int8 KIVI (1 B) | 97.7% | +0.10% | viable 4× tier |
| int4 KIVI (0.5 B) | 92.6% | −1.77% | not viable at a 1% gate |

Resulting theorem (verified, extends rec-kcb): **any reduced-precision store is
discontinuous** — perturbations cross quantization boundaries and amplify by
Δ/ulp regardless of rounding rule; only bit-exact reproduction avoids it. This
motivated the quality-based gate (ppl |Δ| < 1% + top-1 ≥ 97% vs the quantizer's
own reference) replacing logit-matching for reduced tiers.

### Correctness-gate apparatus

- `rec-pp2` two-tier max-rel gates (fp32-KV 1e-4) + production **fp32-KV row**
  codified in `qgate/gate_spec_full_v1.json` (rec-kcb ruling: bf16-KV rejected
  for the full config at 2.74e-2 from position 0, ~+0.35% ppl only)
- Hard rule across 6 builds: existing functions instruction-identical
  (label-normalized) in fp32kv emits — byte-identical md5s verified
- GEPA_SHRINK semantics documented and pinned per variant (SHRINK=1+fence
  promoted for fp32kv full config; SHRINK=0 for int8kv)

## Remaining work

1. **int8-KV → full config** (bead chain, dependency-ordered):
   `rec-5ql` GQA read path (blocker) → `rec-ad6` sliding-window paths (35
   layers) → `rec-czs` full-config determinism recipe → `rec-3tv` correctness
   gates (torch check + int8 quality tier in `gate_spec_full_v1.json`) →
   `rec-7oc` full-config dusk bench row vs 33.40 fp32kv / 26.80 incumbent.
   `rec-776` (LDS/LOP_PF) only if the recipe needs them.
2. **rec-0v1**: root-cause SHRINK=1 + int8 nondeterminism (scratch spill-slot
   reuse is the current suspect; store-drain hypothesis refuted by disasm).
3. **Validation wave** (rec-109): R9700/MI50 correctness portability —
   blocked, host `albion` offline.
4. **Prefill path**: eager-torch INT4 prefill ~750 tok/s vs the 2,462 dense
   floor — a fused prefill kernel is the largest single remaining win.
5. **Long-context validation**: current gate depth 512/32 tokens; target 128k+
   (the int8-KV memory tier exists precisely for this).
6. **Serve-path integration**: OpenAI-compatible HTTP endpoint for the
   megakernel (currently bench-harness only).
7. **Real-Flint cutover decision**: megakernel vs llama.cpp-Vulkan as the
   serving engine on dusk (megakernel leads the mid record; full config at
   1.25× incumbent as of rec-04i).
8. **Compute-precision tiers**: bf16/int8 activation compute trials
   (accumulators stay fp32/int32; gfx1150 has `v_dot2_f32_bf16` and
   `v_dot4_i32_i8`); expected win is prefill + register pressure — decode is
   bandwidth-bound.

## Benchmarks & gates index

- `docs/benchmarks/` — 40 artifact directories, each with README + logs.
  Latest: `rec-ywy-int8kv-mid-dusk-2026-11-24`,
  `rec-hyf-kv-precision-ladder-2026-11-24`,
  `rec04i-lop8-shrink1-fence-depth8192`
- `docs/gates/rec-kcb-full-config-tiers-v1.md` — ratified tier spec
- `qgate/gate_spec_full_v1.json` — production gate rows
- Dusk coordination: benches run under ops_flint cotenant rules
  (display/serving negotiated per window; restore verified same-day).

## Upstream

Forked from [`recursive-oss/rec`](https://github.com/recursive-oss/rec) — the
Apple Silicon engine and its M4-Pro benchmarks are upstream's; everything AMD
(HIP megakernel, gfx1150 port, gates, quantization studies) lives on this
branch. Upstream's install/API docs apply to the Mac lineage only.
