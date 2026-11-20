# rec-xh7 step 0: UMA feasibility and the llama.cpp-Vulkan incumbent row (dusk 890M, 2026-11-20)

Claude correctness lane. Measured on dusk (Radeon 890M, gfx1150; llama.cpp `a37cc46`
build-vulkan, RADV STRIX1) between 12:53Z and 13:22Z, after the 11:46Z boot. No kernel
work was done; the requant-vs-native-quant decision is still with Alex.

Raw data and scripts are in this directory. `summary.json` holds the aggregates; the
`*.jsonl` files hold one line per run, with UTC time, cotenant state, VRAM and gate result.

## 0. What the target actually is (GGUF header, `gguf_meta.py`)

`flint-preview-UD-Q3_K_XL.gguf` (18.49 GB) has this architecture:
- 49 layers;
- 256 experts with top-6 routing (sigmoid, normalized), plus 2 shared experts;
- 2 leading dense layers (FFN 6144);
- 14 full-attention and 35 sliding-attention layers (window 513);
- hidden 2048, 32/4 heads, vocab 150144.

**It is not a Q3_K model.** Tensor bytes by type:

| type | GiB | used for |
|---|---|---|
| IQ3_S | 6.12 | expert gate/up (21 layers), down (15) |
| IQ3_XXS | 4.59 | expert gate/up (24 layers) |
| IQ4_XS | 4.40 | expert down (31 layers), plus 1 layer of attention and shexp |
| Q8_0 | 1.55 | attention (36 layers), shared experts, dense FFN, embeddings, lm_head |
| IQ2_S | 0.16 | expert gate/up, layer 2 |
| Q5_K / Q6_K / Q4_K | 0.34 | attention and shexp in about 12 layers, one down |
| BF16 / F32 | 0.05 | router, norms, sinks |

Scope consequence for decision (a): "in-kernel Q3_K dequant" really means in-kernel
IQ3_S + IQ3_XXS + IQ4_XS + Q8_0. IQ2_S and three K-quants are also needed for a few
tensors. Each format needs a reference (torch/numpy) pinned bit-exact to llama.cpp's
`dequantize_row_*`. The i-quants use codebook grids and sign tables, so this is several
formats, not the single Q3_K block assumed in the 01:29Z correctness-lane estimate. That
estimate is superseded.

## 1. UMA co-residency (`uma_probe.py`, bounded)

### Memory layout

The kernel log shows a 49152 MiB BIOS VRAM carve-out (`amdgpu: VRAM: 49152M`). The OS
manages a separate 47.6 GB (`MemTotal` 46.7 GiB), so the board has 96 GB physically.
GTT is limited to 23.3 GiB, taken from OS RAM. Note: older fleet notes say "VRAM ≈
system-RAM share". That no longer holds as of this boot.

### Resident cotenants (per-process drm fdinfo)

| server | VRAM | GTT |
|---|---|---|
| flint :8080 | 20.8 GiB | 0.45 GiB |
| qwen :8091 | 4.0 GiB | 0.52 GiB |
| E4B :8090 | 3.2 GiB | 2.7 GiB |

### Needs, computed from the real shapes

| item | GiB |
|---|---|
| kernel bank, requantized affine INT4 g64 (4.5 bpw, embed 8.5) | 21.02 (21.16 with an 8-bit lm_head) |
| kernel bank, GGUF quants as-is | 17.22 |
| KV fp32 (validation build) at 8k / 131k / 262k | 0.51 / 7.07 / 14.07 |
| KV bf16 at 8k / 131k / 262k | 0.25 / 3.53 / 7.03 |
| CPU torch fp32 reference: packed weights resident in host RAM | ≈ bank size |

### Probe method

Each probe allocated VRAM in 1 GiB `hipMalloc` + `hipMemset` steps, then, while holding
that, host RAM in 1 GiB touched steps. Stop rules:
- target reached;
- first allocation failure;
- VRAM free (from sysfs) below 1.5 GiB;
- `MemAvailable` below 8 GiB;
- GTT growth above 1.5 GiB.

### Results

| probe | cotenants | VRAM held in carve-out | spill | host RAM held at once | MemAvailable left |
|---|---|---|---|---|---|
| A ladder | flint, E4B, qwen up | **18.0 GiB** | 19th and 20th GiB spilled to GTT | 22 GiB | 11.3 GiB |
| B ladder | qwen stopped | **22.0 GiB** | 23rd GiB spilled to GTT | 22 GiB | 12.9 GiB |
| B1 single 21.5 GiB `hipMalloc` | qwen stopped | 21.5 GiB, all in VRAM | none | — | 34.8 GiB (2.46 GiB VRAM free) |

Everything returned to baseline after each probe. qwen was restarted and healthy.

### Two hazards for any driver on this box

1. **`hipMemGetInfo` is blind to other processes.** It reported 47.8 GiB free while
   the Vulkan llama-servers held 28.8 GiB. A capacity preflight must read sysfs
   `mem_info_vram_used`.
2. **`hipMalloc` past the carve-out silently spills into GTT (host RAM).** It still
   returns success. That is the mechanism behind the earlier hard hangs, so the driver
   should refuse when bank + KV > `vram_total - vram_used - 1.5 GiB`.

### Verdict on (b)

The reduced-layer / discrete-host compromise is **not mandatory** for the rec-pp2
harness cases at ≤ 8k context, under three conditions:

- **Pause llama-qwen during the run.** That gives 22 GiB of VRAM headroom against a
  requant validation build of 21.0 + 0.5 GiB. The single 21.5 GiB allocation was
  measured to fit, with about 1 GiB of carve-out slack after scratch. The GGUF-as-is
  bank (17.7 GiB including KV) fits with 4.3 GiB to spare.
- **Run the reference on the CPU.** It co-resides in host RAM: 22 GiB were held next to
  the VRAM hold with ≥ 11 GiB left. The harness runs reference and kernel as separate
  processes, so they can also run one after the other.
- **Have the driver preflight from sysfs** and refuse to spill.

With all cotenants up, the requant validation build does **not** fit (18.0 GiB
headroom): 3.5 GiB would spill silently. At long context with flint up, nothing fits:
fp32 KV at 131k makes the requant build 28 GiB.

Not measured: the actual peak of a full-config torch fp32 reference (its transients) and
its runtime. No full-config affine checkpoint exists yet; budget about 3 GiB above the
8 GiB floor.

## 2. Incumbent row: llama.cpp-Vulkan decode (hygiene protocol)

### Mid-shaped GGUF

`build_mid_gguf.py` writes the rec mid-prod checkpoint (sha `f494e788`, the rec-pp2
production checkpoint, seed 0) as a flint-arch GGUF:
- metadata and tokenizer are copied from the real GGUF, with the shape keys taken from
  the mid config;
- weights are dequantized with the pinned `torch_rocm.dequant` and written as F16;
- the F16 file was then quantized with `llama-quantize`, using an imatrix from 20 chunks
  of llama.cpp docs text. The imatrix affects numerics only, not speed.

**Fidelity** (`fid_f16.json`): llama.cpp F16 on Vulkan, greedy, against the torch fp32
reference, teacher-forced on llama.cpp's own trajectory:
- 5-token chat prompt: 16/16 tokens match;
- 64-token random prompt: 13/16 match. The 3 flips are near-ties, with reference margins
  of 2.5e-3, 7.6e-3 and 1.9e-3.

### Variants

| file | experts gate/up | experts down | everything else | lm_head | size |
|---|---|---|---|---|---|
| `mid-ud-iq3xxs` (real file's majority type per role) | IQ3_XXS | IQ4_XS | Q8_0 | Q8_0 | 1.19 GB |
| `mid-ud-iq3s` (sensitivity) | IQ3_S | IQ4_XS | Q8_0 | Q8_0 | 1.21 GB |
| `mid-ud-iq3xxs-lmq4` | IQ3_XXS | IQ4_XS | Q8_0 | q4_0 | 1.03 GB |
| `mid-pure-q4_0` (same bytes per token as the megakernel bank) | q4_0 | q4_0 | q4_0 | q4_0 | 0.86 GB |

### Method

Each mid run is `llama-bench -ngl 99 -fa on -p 0 -n 32 -d {512,8192} -r 5`: 32 tokens
generated at depth d, mean of 5 repetitions. Every run was gated on `gpu_busy_percent`
< 5 for 3 samples at 0.3 s, and on 0 busy slots on :8080/:8090/:8091 queried at
`100.98.73.29`. The slot check was repeated after each run; all 32 runs were clean.

The full model was measured through the running :8080 server: an 8192-token random
token-id prompt with `cache_prompt` off, 64 greedy tokens, timings taken from
`/completion`.

Cotenant states:
- **S0:** llama-qwen stopped, flint and E4B resident and idle, untouched;
- **S1:** qwen also resident and idle.

ABBA order over states: S0 S1 S1 S0, then two S0 windows for the lm_head and pure-q4_0
rows.

### Decode tok/s (mean of runs; range across runs)

| row | state | depth 512 | depth 8192 |
|---|---|---|---|
| mid UD mix (iq3xxs) | S0 | 120.11 (118.7–121.7, n=4) | 108.12 (107.1–108.9, n=4) |
| mid UD mix (iq3xxs) | S1 | 120.62 (120.2–121.1, n=2) | 109.90 (109.7–110.1, n=2) |
| mid UD mix (iq3s) | S0 | 120.83 (119.4–122.3, n=2) | 109.09 (108.7–109.5, n=2) |
| mid UD mix (iq3s) | S1 | 121.06 (120.6–121.5, n=2) | 107.58 (107.0–108.2, n=2) |
| mid UD mix, lm_head q4_0 | S0 | 161.49 (160.9–162.0, n=4) | 139.26 (138.8–140.2, n=4) |
| mid pure q4_0 | S0 | **201.87** (201.8–202.0, n=2) | 168.59 (165.1–172.1, n=2) |
| full Flint UD-Q3_K_XL (:8080 server) | S0 | — | **26.56** (26.42, 26.71); prefill ≈ 464 tok/s |
| full Flint UD-Q3_K_XL (:8080 server) | S1 | — | 26.77 (26.69, 26.86) |
| megakernel run-of-record (`d1f6208`, `decode32_after_512` differential) | — | 120.78 (117.1–123.5) | not measured |

Idle-resident cotenants make no measurable difference: S0 vs S1 is within run spread. I
did not drive active cotenant load (the "≈10.2 with cotenants" figure). The "31.1 @ 8k
solo" figure could not be reproduced and its provenance isn't recorded in the repo:
through the server, 8k context measures 26.6–26.8 here.

### Bytes per decode token (estimate)

Computed as parameters touched per token × bits per weight. Not measured.

| row | MB / token | tok/s | effective GB/s |
|---|---|---|---|
| megakernel INT4 | 314.5 | 120.78 | 38.0 |
| llama.cpp UD mix (lm_head Q8_0) | 566.3 | 120.6 | 68.3 |
| llama.cpp UD mix, lm_head q4_0 | 412.6 | 161.5 | 66.6 |
| llama.cpp pure q4_0 | 314.5 | 201.9 | 63.5 |

### Reading

- **Like-for-like on bytes**, at the same mid-8L shape and positions:
  - llama.cpp-Vulkan pure q4_0 runs 202 tok/s against the megakernel's 120.78, so the
    megakernel is **0.60×** the incumbent's speed;
  - it moves about 38 GB/s against llama.cpp's 64–68 GB/s.
- **Against the shipped quant mix**, the megakernel is at parity: 120.8 vs 120.1. That
  parity comes from the UD mix's Q8_0 lm_head, which costs 327 MB per token.
- **The protocols differ.** llama-bench tg32 at depth 512 covers positions 512–543. The
  megakernel differential covers 516–547 and includes argmax; llama-bench includes
  sampling. q4_0 is symmetric while ours is affine, so we do one extra FMA per group.
  None of these is large enough to close a 1.67× gap.
- **Implication for rec-xh7:** scaling the current megakernel to full Flint is not on
  track to beat the incumbent. Its bandwidth efficiency, not the format, is the gap to
  close first.

## Reproduce (on dusk, `~/rec-ve/xh7`)

    PYTHONPATH=~/llama.cpp/gguf-py $PY build_mid_gguf.py ~/models/flint/flint-preview-UD-Q3_K_XL.gguf ~/rec-ve/ckpt-pp2/mid-prod mid-f16.gguf <rec engine/src>
    llama-imatrix -m mid-f16.gguf -f calib.txt -o mid-imatrix.gguf -ngl 99 -c 512 --chunks 20
    llama-quantize --imatrix mid-imatrix.gguf --tensor-type ffn_gate_exps=iq3_xxs --tensor-type ffn_up_exps=iq3_xxs --tensor-type ffn_down_exps=iq4_xs mid-f16.gguf mid-ud-iq3xxs.gguf q8_0 4
    #   iq3s variant: iq3_s for gate/up; lmq4: add --output-tensor-type q4_0; pure: --pure ... q4_0
    ./bench_xh7.sh; ./bench_lm.sh; ./bench_q4.sh      # each restarts llama-qwen on exit
    timeout 300 $PY uma_probe.py <vram_GiB> <host_GiB> <cotenant-tag> [single]   # HSA override unset

GGUF sha256 prefixes:

| file | sha256 |
|---|---|
| mid-f16 | `f186ff17` |
| mid-ud-iq3s | `2046dc63` |
| mid-ud-iq3xxs | `26917351` |
| mid-ud-iq3xxs-lmq4 | `6969dcf2` |
| mid-pure-q4_0 | `c9001692` |

The first run of `bench_xh7.sh` (13:08Z) printed values within about 1% of the main run.
Its jsonl was truncated by a sourcing bug in `bench_lm.sh`, since fixed, so the 13:18Z
rerun in `bench_xh7_main.*` is the run of record.
