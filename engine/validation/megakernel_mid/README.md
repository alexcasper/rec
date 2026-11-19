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
results are in section 6.

Compute must be fp32 (fp32 accumulators and intermediates, with bf16 only for
stored KV). A kernel with bf16 activations end to end lands at ~1.5e-2 to
2.2e-1 with 1–3 token flips beyond the near-tie margin. That is the measured
`torch-bf16` stand-in, which is itself more faithful than MLX bf16. Such a
kernel cannot meet this target, which is why the gate is set where it is.
This is stricter than `docs/rocm-model-layer.md` §3 ("inter-op activations
bf16"). If Phase B needs bf16 activations for speed, keep an fp32 validation
build, as the tiny kernel does with float shared buffers.

### 4a. MoE router near-ties: PROPOSED amendment, NOT in force

**Status.** This section is proposed and awaits coordinator adoption. Until
it is adopted, the gate in section 4 applies unchanged: a router near-tie
flip is a FAIL. The checker does not implement this amendment yet.

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

**Proposed wording (replaces nothing; adds a rule to section 4):**

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
