# Mid-8L megakernel validation target (rec-pp2)

This is the correctness reference for a Flint **mid-8L** decode megakernel
(Gemini Phase B, branch `rec/gemini-kernels`) on dusk (Radeon 890M, gfx1150).
Owner: Claude arm (correctness lane). Kernel sources are not touched here.

The kernel is judged against the **torch model layer** (`rec_mlx.torch_rocm`,
the `REC_BACKEND=rocm` path). That layer is pinned to the MLX production
model by `engine/tests/test_torch_rocm.py` (fp32 logits 1.6e-6) and to
production engine behaviour by `engine/tests/rocm_engine_e2e.py`. The
reference runs **fp32 compute** and is **teacher-forced along the kernel's
own tokens**, so a near-tie flip cannot cascade.

## 1. Checkpoint (production format)

```bash
python -m rec_mlx.synthetic --preset mid --format production --out ~/rec-ve/ckpt-pp2/mid-prod
```

The config is mid-8L: hidden 2048, 32/4 heads, head_dim 128, MoE 32 experts
(top-2), moe_inter 512, shared 1024, vocab 150144. Layer 0 is dense, layers
alternate sliding/full, and sinks are on. The tensors match the
`RecursiveSI/flint-int4` export: **U8** packed INT4 weights `[..., out, K/2]`,
**BF16** scales/biases/norms/router/sinks, an INT8 embedding, and **no
`expert_bias`**. See `docs/rocm-model-layer.md` for the device layout
contract.

The build is deterministic. model-00001 sha256 is
`f494e7885d10bdedf5d885faef826a27fdbe4bfb1fb11781fac12c7cabbd92ad` on both
dusk and dawn, and kernel outputs are bound to it. The tokenizer uses the
post-92f9bc2 fix, so every id is < vocab.

## 2. Cases

`python mid_mk_check.py cases --ckpt CK` prints JSON for drivers in any language.

| case | prompt | n_out | exercises |
|---|---|---|---|
| `short4` | bos + 3 tokens | 24 | first-window decode |
| `chat` | bos, `<\|user\|>`, 2 tokens, `<\|assistant\|>` | 24 | special-token ids (3..5) |
| `wrap512` | bos + 511 seeded tokens | 24 | positions 0..535 > 513: **sliding ring wrap** (`slot = pos % 513`), long full-attention KV |

## 3. Kernel output (`rec-mk-out/1` npz)

| key | type | meaning |
|---|---|---|
| `schema` | str | `"rec-mk-out/1"` |
| `case` | str | case name |
| `ckpt_sha256` | str | sha256 of `model-00001.safetensors` the kernel ran on |
| `impl` | str | kernel name/build (free text) |
| `kv_dtype` | str | `"fp32"` or `"bf16"`: KV-cache storage precision of this build |
| `prompt` | int32 [P] | must equal the case prompt |
| `tokens` | int32 [n_out] | greedy tokens; `tokens[i]` chosen from `step_logits[i]` |
| `step_logits` | f32 [n_out, V] | logits that picked each token (**required**) |
| `last_logits` | f32 [V] | logits after consuming `tokens[-1]`, the same convention as the tiny kernel's `out_last_logits` |

**Driver rule from rec-ju9 / rec-hjr.** `ctypes.CDLL` the kernel `.so`
**before** importing torch, because torch's bundled libhsa preempts the loader
and breaks newer HSA symbols. Better still, write the npz from a torch-free
driver process and run the checker as a second process, as the tiny
`tiny_mk_driver.py` does.

## 4. Pass criteria

A run **PASSES** when all of these hold:

1. **Schema and binding.** All keys are present, the prompt equals the case
   prompt, `ckpt_sha256` matches, and every value is finite with ids < vocab.
2. **Token-exact vs the fp32 reference.** Teacher-forced, every `tokens[i]`
   equals the reference argmax at step i. The only exception is a near-tie
   with fp32 margin ≤ **5e-3**, the same rule as the tiny matrix. Accepted ties
   are listed in the report.
3. **Logits.** `max|kernel − ref| / max|ref|` over `last_logits` **and every
   `step_logits` row** must not exceed the tier tolerance:

| declared `kv_dtype` | reference | tolerance | measured floor (correct impl, mid, dusk) |
|---|---|---|---|
| `fp32` | fp32 compute, fp32 KV | **1e-4** | 3.4e-6 |
| `bf16` | fp32 compute, K/V rounded to bf16 at the storage points (K after k_norm+RoPE, V after projection) | **1e-3** | 5.0e-4 |

**Sign-off (adopted coordinator policy).** A production kernel signs off
through its **fp32-intermediate validation build**. That build must PASS all
3 cases in **both** KV tiers: once storing KV as fp32 (gate 1e-4) and once as
bf16 (gate 1e-3). A direct run of the bf16-activation production build is
**not** a substitute for sign-off. The fp32 tier is not optional. On mid-8L the `softmax-mix` MoE fault (mixing
weights softmax instead of renormalised sigmoid) keeps tokens exact. Against
the bf16-KV reference it lands at 4.2e-4 to 9.0e-4: under 1e-3 and the same
size as the correct bf16-KV floor (5.0e-4), so the bf16 tier cannot see it.
Against the fp32 reference it is 2.5e-4 to 1.05e-3 (`step_logits`), at least
2.5× over the 1e-4 gate and about 70× over the correct fp32 floor. Measured
results are in section 6. **Reference device (rec-s3m, adopted 2026-11-20):** the sign-off reference is torch fp32 on
the **CPU** (`mid_mk_check.py` default, `SIGNOFF_REF_DEVICE`), the same reference rec-ft9 adopted
for the engine e2e. Every matrix line records `ref=torch-fp32@<device>`; a matrix with any other
reference device is diagnostic only and exits 2 even when it passes. Reason: the bf16-KV tier is
reference-device sensitive. On albion the same gfx1201 kernel outputs score short4 bf16-KV
9.24e-4 (PASS) against the CPU reference but 1.056e-3 (FAIL) against a torch 2.10/ROCm 7 R9700
reference, and dusk's d1f6208 run-of-record outputs fail that cell on the R9700 reference too
(1.096e-3); the fp32 tier is unaffected (<= 3.3e-6).

Compute must be fp32 (fp32 accumulators and intermediates, with bf16 only for
stored KV). A kernel with bf16 activations end to end lands at ~1.5e-2 to
2.2e-1 with 1–3 token flips beyond the near-tie margin. That is the measured
`torch-bf16` stand-in, which is itself more faithful than MLX bf16. Such a
kernel cannot meet this target, which is why the gate is set where it is.
This is stricter than `docs/rocm-model-layer.md` §3 ("inter-op activations
bf16"). If Phase B needs bf16 activations for speed, keep an fp32 validation
build, as the tiny kernel does with float shared buffers.

### 4a. MoE router near-ties (ADOPTED 2026-11-19, threshold PROVISIONAL)

**Status.** The coordinator adopted this rule on 2026-11-19, and the checker
implements it (`router_substitution` in `mid_mk_check.py`). It is the
**interim** gate. The long-term fix is exact routing teacher-forcing from
kernel-emitted `step_experts` (Gemini lane). **ROUTER_TIE_TOL = 2.5e-4 is
PROVISIONAL.** Recalibrating it across cases and seeds is the required bead
rec-tan.3. The stand-in calibration is in section 4b. The kernel calibration
(section 4c) confirms 2.5e-4 and proposes a declared-routing count rule.

**Preferred path.** If the kernel output carries `step_experts`, the checker
takes its routing deviations from the kernel's declared routing and searches
nothing (`router_path: "step_experts"`). It applies the all-declared rule
(section 4c, adopted under rec-tan.4). The single-swap search below is the
fallback for outputs without `step_experts`.

**Why it is needed.** The near-tie rule in section 4 (fp32 margin ≤ 5e-3)
covers only the output-token argmax. The other discrete decision in the
model, MoE top-k expert selection, is not teacher-forced. At the bf16-KV
tier, a correct implementation perturbs the selection scores by roughly
1e-4. That is enough to flip an expert sitting on the top-k boundary.
Everything downstream of the flip then fails, even though nothing is wrong.

Evidence (rec-tan, kernel 045c72f, `chat`, fp32 compute + bf16 KV, dusk; see
`results-rec-tan-*` and `router_tie_probe.py`):

- **Stock check.** Step 15 fails with step_rel 2.04e-1 and last_rel 1.37e-2.
  The token differs from the reference at an fp32 margin of **8.1e-2**, 16×
  the tie tolerance, so this is not an output-token tie.
- **Reference router at step 15.** In layer 2, the 2nd and 3rd selection
  scores (experts 12 and 7) differ by **4.1e-5**.
- **Forced replay.** Swapping the reference's layer-2 position-19 choice from
  12 to 7, and changing nothing else, gives step_rel **6.87e-4**, last_rel
  4.17e-4 and **24/24 tokens exact**. That is inside the 1e-3 tier at the
  correct floor.
- **Controls.** Swapping the next two closest boundaries (gaps 1.8e-4 and
  2.4e-4) leaves the failure at 2.0e-1. Gaps of the same size elsewhere
  (`wrap512` L2 3.9e-5, L6 5.0e-5) did not flip. Flips are stochastic, so the
  gate is currently a coin toss for any correct bf16-KV build.
- **fp32-KV tier.** The same kernel passes 3/3 at ≤ 3.65e-6. Its smallest
  observed boundary gap is 6.3e-5, about 17× that tier's logit floor
  (3.65e-6). Its semantics are validated by that tier.

**Rule in force (adds to section 4):**

> **Router near-tie substitution (bf16-KV tier only).** If a run fails the
> bf16-KV tier, the checker may re-run the teacher-forced reference with at
> most **one** MoE routing substitution per case, under all of these
> conditions:
>
> 1. The substitution swaps the k-th and (k+1)-th expert at a single
>    (layer, position).
> 2. At that (layer, position), the reference's selection-score gap
>    (sigmoid + expert_bias, between rank k and rank k+1) is
>    ≤ **ROUTER_TIE_TOL = 2.5e-4** (absolute, in selection-score units).
> 3. The position lies at or before the first failing step.
>
> The run PASSES only if the substituted reference meets the **full**
> unchanged bf16-KV criteria: tokens exact (output ties ≤ 5e-3), and
> last_logits and every step_logits row ≤ 1e-3. The report must list the
> substitution (layer, position, experts, gap). It counts as a near-tie, like
> accepted output ties.
>
> - The **fp32-KV tier gets no router allowance.** Sign-off still requires
>   that tier to pass outright, so a selection or mixing bug cannot hide
>   behind this rule. A selection bug changes routing at non-tie positions and
>   fails the fp32 tier directly; the softmax-mix fault is caught there at
>   about 70× its floor (section 6).
> - More than one substitution, or one at a gap > ROUTER_TIE_TOL, is a FAIL.
> - A bf16-activation production run remains no substitute for sign-off.

**Checker behaviour.**
- The rule runs only when the bf16-KV tier fails on tokens or logits; schema,
  shape and non-finite failures are not rescued.
- Candidates are the reference's boundary gaps ≤ ROUTER_TIE_TOL at positions
  ≤ the first failing row (a bad token or a row over the tier; last_logits
  counts as row n). They are tried smallest gap first, at most
  `MAX_ROUTER_CANDIDATES` = 8, each as a **single** swap.
- The first swap whose substituted reference meets the full criteria is
  accepted. It is reported as `router_near_tie`, appended to
  `near_ties_accepted`, and shown in the `matrix` line, with the stock failure
  kept in `stock_problems`.

**Matrix coverage.** Per the adopted policy, `matrix` now exits 0 only if
**every case passes in both KV tiers**; it reports missing (case, tier) pairs.

**Calibration caveat.** ROUTER_TIE_TOL rests on one observed flip (gap 4.1e-5)
and on boundary gaps up to 2.4e-4 that did not flip. It is a heuristic about
6× the observed flip, kept below the bf16-KV logit floor (5.0e-4, a different
unit). Before or with adoption, recalibrate it: produce the torch-fp32-bf16kv
stand-in in streaming order on all 3 cases and several prompt seeds, and
record the largest gap that flips.

**Recommended follow-up (strengthens the rule, not required for it).** The
kernel could emit `step_experts` (int32 [P + n_out, layers, k], its chosen
expert ids) as an optional `rec-mk-out/1` key. The checker could then
teacher-force routing exactly, instead of searching for a substitution, and
apply the same gap and count limits to the positions where it differs.

### 4b. ROUTER_TIE_TOL calibration (rec-tan.3, 2026-11-19)

**Setup.** `router_tie_calibrate.py` ran on dusk (mid-prod `f494e788…`, top-2,
7 MoE layers) over the 3 cases plus 10 seeded prompts. Odd seeds are 32
tokens; even seeds are 512 tokens, which cross the sliding window. Each
prompt decodes 24 tokens. The implementation under test is a **correct**
stand-in in kernel order (prefill, then one token per step) in both tiers.
It is compared with the batch reference teacher-forced on the stand-in's
tokens.

**Measured quantity.** The relevant noise is
|Δgap| = |(r_k − r_k+1) − (s_k − s_k+1)|: how far the implementation moves
the reference's top-k boundary. A flip happens when |Δgap| exceeds the gap.
Boundaries after a run's first flip are excluded, because they are
downstream of it. Results are in `results-rec-tan3-calibration.json`.

| tier | boundaries | \|Δgap\| median | p99 | p99.9 | max | > 2.5e-4 | > 5e-4 | runs failing stock | runs failing with rule |
|---|---|---|---|---|---|---|---|---|---|
| bf16-KV | 21,406 | 4.2e-6 | 7.1e-5 | 1.6e-4 | **2.9e-4** | 3 | 0 | 1 / 13 | **0 / 13** |
| fp32-KV | 24,871 | 1.2e-7 | 1.0e-6 | 1.6e-6 | 3.0e-6 | 0 | 0 | 0 / 13 | 0 / 13 |

**Natural flips on correct implementations.**
- `seed4-len512`, L4 pos 369, gap 3.5e-6: the stock gate FAILs. The rule
  passes it by swapping exactly that boundary; the L5 flips that follow are
  downstream of it.
- `seed6-len512`, L4 pos 208, gap 2.2e-6: harmless, and the stock gate passes.
- Kernel 045c72f `chat`, L2 pos 19, gap 4.1e-5 (section 4a).

Every observed flip sits at a gap ≤ 4.1e-5. The fp32 tier's |Δgap| is two
orders of magnitude smaller, with no flips, so it needs no allowance.

**Reading.**
- Without the rule, the stock gate falsely fails a correct bf16-KV
  implementation about 1 time in 13 here (the stand-in), and 1 in 3 for the
  kernel.
- With the rule, it fails 0 of 13.
- 2.5e-4 covers every observed flip with a 6× margin. However, it sits
  **below** the tail of the noise: 3 of 21,406 boundaries move by more than
  2.5e-4, up to 2.9e-4. A correct build can therefore, rarely, flip a
  boundary with a gap in (2.5e-4, 2.9e-4] and be falsely failed.
- No boundary moved by more than 5e-4.

**Recommendation: superseded by section 4c.** This section originally
suggested 5e-4, from the stand-in's tail. Kernel data confirms 2.5e-4, so
that suggestion is withdrawn. The rule's safety does not rest on the
threshold: a swap counts only if the full unchanged criteria then pass, and
the fp32 tier has no allowance.

**Limitation.** This is the stand-in's noise, not the kernel's. The kernel
reduces in a different order. Once the kernel emits `step_experts`, measure
the kernel's own |Δgap| (or switch to exact routing teacher-forcing) and
re-check the threshold.

### 4c. Kernel routing calibration from `step_experts` (rec-tan.3, 2026-11-19)

**Setup.**
- **Kernel:** Gemini 4c1cb9a, `flint_mid.so` on dusk, workers 16.
  `router_kernel_emit.py` runs it torch-free and writes `step_experts` for the
  same 3 cases and 10 seeds as section 4b, in both validation builds
  (fp32-KV mode 2 and fp32-compute/bf16-KV mode 1). Each run decodes 24
  tokens.
- **Analysis:** `router_kernel_analyze.py` forces the reference to the
  kernel's routing everywhere (`Reference.run(force=…)`). It then compares the
  kernel's expert set with the reference's own top-k at every MoE
  (layer, position). Upstream routing is identical by construction, so each
  deviation is a clean boundary crossing, not a downstream consequence of
  another one.
- **Limitation:** the kernel emits expert ids, not scores, so its |Δgap| (the
  section 4b quantity) cannot be measured. The table bins the reference gap at
  every boundary and counts how many of those the kernel crossed.
- Results: `results-rec-tan3-kernel-calibration.json`.

| tier | boundaries | crossings | non-k/(k+1) | crossing gap median / max | < 1e-5 | 1e-5–5e-5 | 5e-5–1e-4 | 1e-4–2.5e-4 | 2.5e-4–5e-4 | 5e-4–1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16-KV | 24,871 | 22 | 0 | 1.4e-5 / **9.5e-5** | 3 / 19 | 17 / 70 | 2 / 76 | **0 / 233** | **0 / 349** | 0 / 673 |
| fp32-KV | 24,871 | **0** | 0 | — | 0 / 22 | 0 / 65 | 0 / 98 | 0 / 214 | 0 / 334 | 0 / 680 |

Bin columns are crossings / boundaries in that gap bin. No boundary with a
gap above 1e-3 was crossed in either tier.

| tier | runs failing stock | runs failing adopted rule (≤ 1 deviation) | runs failing proposed rule (below) | reference forced to kernel routing: step / last rel max, tokens |
|---|---|---|---|---|
| bf16-KV | 6 / 13 | **5 / 13** (seed2, 4, 5, 6, 10: 2–5 deviations each) | **0 / 13** | 6.87e-4 / 5.63e-4, all exact |
| fp32-KV | 0 / 13 | 0 / 13 | 0 / 13 | 3.80e-6 / 3.02e-6, all exact |

The 3 sign-off cases pass the formal `check` in both tiers under the adopted
rule. Section 6's chat swap reappears here as the kernel's only chat
deviation (L2 pos 19, gap 4.1e-5). `wrap512` bf16-KV has 3 harmless
deviations and passes stock.

**Threshold: 2.5e-4 CONFIRMED from kernel data.**
- The kernel's crossings stop at 9.5e-5, a 2.6× margin under 2.5e-4.
- None of the 582 boundaries with gaps in 1e-4 to 5e-4 was crossed.
- The crossing rate is 16–24% below 5e-5, 2.6% in 5e-5 to 1e-4, and 0 above.
- The section 4b suggestion of 5e-4 is **withdrawn**. It came from the
  stand-in's |Δgap| tail, which is a different implementation; the kernel's
  crossings are the direct evidence.

**ADOPTED 2026-11-19 (rec-tan.4): the count limit was the real problem.**
- On the adopted rule's count limit, a correct kernel fails 5 of 13 bf16-KV
  prompts. Each of those has 2–5 genuine near-tie crossings, all at gaps
  ≤ 9.5e-5, and with the kernel's routing forced each one sits inside the
  tier.
- The single-swap limit was written for **search**, where every extra swap
  is another free hypothesis. When the kernel **declares** its routing,
  nothing is searched, and each declared deviation can be checked
  individually.

Rule in force for the `step_experts` path (`ROUTER_EXPERTS_POLICY =
"all-declared"`, the default since rec-tan.4; `"single"` stays selectable for
comparison only):

> **Declared-routing rule (when `step_experts` is present).**
>
> - **bf16-KV tier:** every (layer, position) where the kernel's experts
>   differ from the reference's own top-k must satisfy both of:
>   (1) it is a k-th/(k+1)-th exchange;
>   (2) its gap is ≤ ROUTER_TIE_TOL.
>   Deviations are measured on the reference forced to the kernel's routing.
>   The run then PASSES only if the full unchanged section-4 criteria hold
>   against that forced reference.
> - **fp32-KV tier:** any declared deviation is a FAIL, even if the stock
>   check passes. This tightens the adopted rule; the kernel showed 0 of
>   24,871.
> - Without `step_experts`, the adopted single-swap search rule applies
>   unchanged.

The safety argument is the same as before:
- deviations are possible only at genuine near-ties (gap ≤ 2.5e-4, about
  1.6% of boundaries);
- misreported routing cannot pass, because the forced reference must still
  reproduce the kernel's logits (tested);
- the fp32 tier is stricter than before.

## 5. Commands (dusk)

```bash
source ~/rec-ve/e2e-env.sh          # NixOS LD_LIBRARY_PATH + HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH=<tree>/engine/src
H=<tree>/engine/validation/megakernel_mid/mid_mk_check.py
CK=~/rec-ve/ckpt-pp2/mid-prod

$PY $H cases --ckpt $CK > cases.json
# kernel driver writes out/<case>_<build>.npz per section 3, then:
$PY $H check  --ckpt $CK out/wrap512_mybuild.npz        # one file, JSON report, exit 0 = PASS
$PY $H matrix --ckpt $CK out/                           # all files, exit 0 = all PASS + all cases covered

# stand-ins / negative controls (no kernel needed):
$PY $H produce --ckpt $CK --case wrap512 --impl torch-fp32-bf16kv --device cuda --out out/x.npz
$PY $H produce --ckpt $CK --case wrap512 --impl torch-fp32 --fault softmax-mix --device cuda --out out/f.npz
```

The checker's reference defaults to `cuda` when available. fp32 on the dusk
GPU equals CPU fp32 to 2.1e-6. Use `--ref-device cpu` to rule out the device.
`--ref-kv` overrides the declared `kv_dtype`. Do not use it to pass a build:
a bf16-KV build that declares fp32 fails, by design and by test.

dusk also serves flint on `:8080` and qwen on `:8091`. This harness never
touches those units or ports.

## 6. Validation run (dusk 890M, 2026-11-18)

These are stand-in producers running the torch layer as if it were the
kernel: prefill, then one token per step, with KV in a 513-slot ring. They
run on the GPU (torch 2.9.1+rocm6.4, `~/rec-ve/.venv-e2e`). The checker runs
on the GPU in fp32. Checkpoint `f494e788…`. The full run took 350 s plus
180 s; each producer takes about 20 s and the checker about 1.5 s per file.
Values are worst-case rel err over the 3 cases (short4 / chat / wrap512), as
`last_logits` / `step_logits`.

| producer (declared kv) | tier | last rel | step rel | tokens | verdict |
|---|---|---|---|---|---|
| torch-fp32 (fp32) | 1e-4 | ≤ 2.6e-6 | ≤ 3.4e-6 | exact ×3 | **PASS ×3** |
| torch-fp32-bf16kv (bf16) | 1e-3 | ≤ 3.2e-4 | ≤ 5.0e-4 | exact ×3 | **PASS ×3** |
| torch-bf16, bf16 activations (bf16) | 1e-3 | 1.5e-2 – 1.2e-1 | 1.6e-1 – 2.2e-1 | 1–3 flips beyond tie | FAIL ×3 |
| fault no-silu (fp32 / bf16) | both | 0.74 – 1.04 | 0.92 – 1.09 | 22–24 wrong | FAIL ×6 |
| fault softmax-mix (fp32) | 1e-4 | 9.4e-5 – 7.3e-4 | **2.5e-4 – 1.05e-3** | exact ×3 | **FAIL ×3** (caught) |
| fault softmax-mix (bf16) | 1e-3 | 4.2e-4 – 7.5e-4 | 7.9e-4 – 9.0e-4 | exact ×3 | PASS ×3 (known blind spot: why sign-off needs the fp32 tier) |

In `chat`, softmax-mix is caught only through `step_logits` (step 2.5e-4,
last 9.4e-5), which is why `step_logits` is required. The `wrap512` case
crosses the 513-token window, so the ring-wrap path is exercised and passes
at the floor values above.

Reproduce with `run_mid.sh` and `run_mid2.sh` (in `~/rec-ve/pp2` on dusk):
build the ckpt, emit cases, run 21 producers, then `matrix`.

## 7. Tests

`engine/tests/test_mid_mk_harness.py` runs on CPU with tiny in production
format. It checks that correct implementations pass in both tiers, that
`no-silu` and `softmax-mix` fail, that a bf16-KV build declaring fp32 fails,
that a wrong ckpt sha or prompt is rejected, and that case prompts are valid
for any vocab.
