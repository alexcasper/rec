# rec-1f4 quality gate v1 (Track B: relaxed kernels)

Ruling (Alex, 2026-11-21, via rec-9t8): Track B may change the megakernel's arithmetic. A relaxed kernel no
longer has to be byte-identical to 2fd83cb; it has to pass THIS gate. The gate is committed before any
relaxed kernel runs, and the eval set is frozen, so every later comparison is against the same items.

Anchors:
- **Reference arithmetic / attribution:** 2fd83cb (the byte-identical Track A line; 24f167d is
  byte-identical to it). The re-baseline (rec-1f4 Step 1) scores 2fd83cb AND 24f167d on this gate. They must
  produce identical reports; that doubles as a byte-identity check of 24f167d on 32 more items.
- **Timing baseline:** 24f167d (dusk ~207-209 tok/s short ctx).
- **Correctness reference:** `rec_mlx.torch_rocm`, fp32 compute, CPU, teacher-forced along the kernel's own
  trajectory: the spec (a) sign-off pin of f0698a5. The reference host and torch version are recorded in
  every report.

## Files

| file | role |
|---|---|
| `eval_set_v1.json` | FROZEN eval set; `sha256` = canonical hash of its items. Never regenerated (the generator refuses to overwrite). |
| `make_eval_set.py` | how it was generated (seed 20261121), kept for provenance |
| `gate_spec_v1.json` | thresholds; pins the eval-set sha256 |
| `qgate.py` | `emit` (dusk GPU) -> `score` (CPU reference) -> `compare` (candidate vs baseline) |
| `qgate_emit.sh` | dusk wrapper: busy gate, cotenant log, kernel-ring check |

## Eval set v1

The dusk-runnable checkpoint is `mid-prod` = `synthetic-mid-flint-seed0-prod` (random-init mid-8L Flint
geometry, synthetic tokenizer; shard sha256 f494e788...). There is no natural-language text for it, so
in-distribution items are **sampled from the fp32 reference model itself** (T = 1).

- **gen (32 items, greedy, n_out = 24):** `legacy/short4`, `legacy/chat` (the rec-pp2 cases, continuity with
  every recorded matrix), `chat/00-09` (bos, user, 2-10 plain tokens, assistant), `plain/00-09` (bos + 1-15
  uniform tokens), `indist/00-09` (bos + 6-15 reference-sampled tokens). 768 decode decisions per build.
- **ppl (8 items):** bos + 31 reference-sampled tokens, so 248 teacher-forced predictions.
- **sha256 72209796906e20dbbb7e69619f98f8b9fae0817b050d35d0f4f82e256915f464** (commit 16b2c42, Step 0a, committed
  before the gate definition and before any relaxed kernel).
- Measured while generating: the synthetic model's next-token entropy is 11.753-11.757 nats against
  11.919 for uniform. The distribution is near-uniform, so "in-distribution" sequences are close to random
  and top1-top2 margins are small. That makes T3 near-tie heavy and T4 a log-prob fidelity check.
- **dusk-safe:** every launch is <= 40 positions (~0.21 s); `qgate.py emit` aborts on any launch > 0.25 s.
  ppl is measured with one-token launches over every prefix (527 positions per item), so no kernel ABI change
  is needed and any variant can be scored as-is.

## Tiers (gate_spec_v1.json)

Builds: `fp32kv` (mode 2, fp32 compute + fp32 KV), `fp32_bf16kv` (mode 1), `bf16kv` (mode 0, production).

| tier | what | rule | kind |
|---|---|---|---|
| **T1** | spec (a), `fp32kv` vs fp32-KV reference, every gen item | max rel logit err <= 1e-4; tokens exact except near-ties (ref margin <= 5e-3); NO routing deviation | **absolute** (spec (a) unchanged) |
| **T2** | spec (a), `fp32_bf16kv` vs bf16-KV reference (all-declared router policy) | **no per-item regression**: every item the baseline passes must pass; an item the baseline fails keeps step rel <= 1.5 x baseline and gains no new problem | relative (see finding 1) |
| **T3** | `bf16kv` production vs fp32-compute bf16-KV reference **forced to the kernel's own routing** (step_experts) | routing deviations <= ceil(1.25 x baseline) + 2; max deviation gap <= max(2 x baseline, 2.5e-4); forced-routing step rel max <= 1.5 x baseline; forced-routing hard disagreements (ref margin > 5e-3) <= baseline; forced-routing token agreement >= baseline - 0.005. Reported: margin histograms (all steps / at disagreements), routing-gap histogram | relative to the re-baseline (finding 2) |
| **T4** | teacher-forced ppl on the 8 ppl items | `fp32kv`: \|ppl/ppl_ref - 1\| <= 1e-4. `bf16kv`: <= max(1.5 x baseline's, 1e-4) | absolute (fp32) / relative (bf16) |
| info | free-running token trajectories vs the baseline kernel | items identical, first divergence steps | not gated |

PASS = every gate in `compare` passes. T1 and T2 keep spec (a) exactly as signed off. The production
bf16 build was never inside the 1e-3 tier (inherent ~3.9e-3), so it is gated relative to the re-baseline
instead of by an absolute tier.

## Re-baseline on this gate (2fd83cb; 24f167d emits byte-identical npz on all 98 outputs)

Reference host: dawn (x86, torch 2.15.0 CPU, 12 threads). Kernel host: dusk 890M, 16 workers, longest
launch 0.225 s (2fd83cb) / 0.197 s (24f167d).

| build | spec (a) | step rel max | forced-routing rel max | routing dev (max gap) | forced tf disagreements | ppl rel vs ref |
|---|---|---|---|---|---|---|
| fp32kv | **32/32** | 1.231e-6 | 1.231e-6 | 0 | 0/768 | -2.76e-8 (max \|dlogp\| 1.5e-6) |
| fp32_bf16kv | **31/32** | 1.053e-3 | 1.053e-3 | 4 (1.13e-4), all k/(k+1) | 1/768, hard 0 | - |
| bf16kv | 0/32 (not tier-gated) | 0.2306 | **4.362e-3** | **92 (1.74e-3)**, all k/(k+1) | 1/768, hard 0 | -6.88e-4 (max \|dlogp\| 0.27) |

Findings the 2-case matrix could not show:
1. **The signed-off arithmetic is marginal on the bf16-KV tier:** 2fd83cb fails spec (a) on `plain/00`
   (1.053e-3 vs 1e-3; no token or routing problem). T2 therefore gates non-regression instead of the absolute
   tier. Whether to recalibrate the tier is Alex's call.
2. **The production bf16 build flips MoE routing on every item** (92 k/(k+1) exchanges over 32 items, gaps up
   to 1.74e-3, well past the 2.5e-4 router-tie rule). Against the unforced reference each flip shows up as
   0.1-0.23 logit error. With the reference forced to the kernel's routing, the arithmetic error is the known
   inherent ~4.4e-3, with 0 hard disagreements. T3 gates routing and arithmetic separately so a relaxed
   kernel's regression is attributable.

## Gate self-test (both directions)

- **Baseline vs itself:** PASS on all 10 gates.
- **Negative control** (`selftest_negative.py`: one fp32kv logit +3e-4 x absmax in chat/03; plain/04 bf16kv rows
  x (1 + 2e-2); every fp32kv ppl logprob + 2e-4): **FAIL on exactly the three targeted gates**. T1 31/32
  (chat/03, 3.0e-4 > 1e-4); T3 forced-routing rel 2.17e-2 > 1.5 x 4.36e-3; T4 fp32kv ppl -2.0e-4 > 1e-4.
  The other 7 gates pass.

Records: `docs/benchmarks/1f4-qgate-2026-11-21/`.

## Limits (be honest about what this gate can and cannot see)

- The checkpoint is synthetic. T4 "ppl" measures numeric fidelity to the reference arithmetic on
  in-distribution sequences, **not language quality**. The real-quality ppl row (full Flint, flint-int4,
  real text) needs the full-config megakernel and is **albion-gated**. So are wrap512 / ctx512 and the R9700
  rows (runbook 5efd0c0).
- Short context only (<= 40 positions): the sliding-window ring wrap and long full-attention KV are not
  exercised on dusk. wrap512 stays in the spec (a) matrix on albion.
- The CPU reference runs off-dusk (dusk has ~8 GB free next to flint/E4B; an fp32 mid-8L reference is
  ~6.5 GB, and an OOM there would hit a cotenant). The reference host is recorded in each report.

## Running

    # dusk (GPU): emit for each kernel .so
    SOS="base_2fd83cb=/path/2fd83cb.so b7_24f167d=/path/24f167d.so" ./qgate_emit.sh
    # CPU host (reference): score each emit dir, then compare
    PYTHONPATH=engine/src python qgate.py score --ckpt mid-prod --dir emit/b7_24f167d --out b7.json
    python qgate.py compare --cand cand.json --base b7.json
