# rec-ywy v1: int8 KV mode in the megakernel, mid-8L bring-up on dusk (2026-11-24), source f97c59c

**Mode 3** (`mid_generate_fp32_int8kv`) is fp32 compute with per-token-head symmetric int8 KV. Each 128-dim K or V row gets one fp32
scale, `absmax/127`, set at KV-write time and RNE-quantized. Reads dequantize: in the LOP path the scale multiplies the loaded K/V
values; in the split and 1-warp paths the K scale multiplies the reduced score and the V scale the softmax weight. The int8 buffers are
allocated lazily on the first mode-3 run, so fp32/bf16 sessions' VRAM is unchanged. Read paths implemented: 1-warp (sinks), split
non-LOP, split LOP. GQA, LDS and LOP_PF are compile-time rejected. (The f97c59c message says the K scale is folded after the reduction
everywhere; in the LOP path it isn't.)

**Hard rule:** in 6 builds (default, split, ctl6, DOT2, full lop8f, full default), every existing function is instruction-identical to
the previous HEAD after label normalization. The fp32kv emit from an int8-containing library is byte-identical to the iteration-6 ctl6
baseline (md5 14814c8311e6).

## Determinism and the recipe pin
| build | depth8192 int8kv fresh sessions | distinct md5s |
|---|---|---|
| SHRINK=1 (mid default) | 5 | 5 |
| SHRINK=1 + rec-11b tail fence at mid | 4 | 4 |
| **SHRINK=0** | 6 | **1** (4598318faded) |

**v1 recipe = build the int8-KV library with `-DGEPA_SHRINK=0`.** The other kernels in the same .so stay deterministic at SHRINK=1.
sunny's writer-side store-drain hypothesis (recywy-int8kv-nondeterminism-isa-answer.md) is refuted by the disasm (sunny f2bdc47):
- the int8 stores are packed into one `global_store_b32` per lane;
- `s_waitcnt_vscnt null, 0` precedes the arrival atomic.

Root cause is open as rec-0v1 (scratch spill-slot reuse is the next suspect).

## Implementation check vs torch (dawn, mid-prod, eval_set_full_v1 teacher-forced; score_i8.json)
| | top-1 agreement | ppl rel | max \|Δlogp\| |
|---|---|---|---|
| kernel fp32kv vs torch fp32 | 100% | 1e-8 | 2.2e-6 |
| kernel int8kv vs torch int8 (same quantizer), 3 read paths | 99.5–99.7% | −1.4e-4..−1.7e-4 | 0.093 |
| torch int8 vs torch fp32 | 90.2% | −0.10% | — |
| kernel int8kv vs torch fp32 (gate comparison) | 89.9% | −0.11% | — |

The kernel reproduces the torch int8 reference: residual differences are rounding-boundary flips, far below the int8-vs-fp32 effect.
**The top-1 ≥ 97% gate can't be judged at mid.** mid-8L is an 8-layer truncation (ppl ~1.8e5, near-flat distributions), and torch's own
int8 scores only 90.2% there. On full Flint, torch int8 per-token-head is 97.7% (rec-hyf). The binding gate runs at full config.

## Speed (mid-8L, lop8 depth path, ABBA 12/12 per row, interleaved; per-row cotenant gate)
| mode / build | ctx8192 tok/s | short tok/s |
|---|---|---|
| fp32kv, SHRINK=1 (run-of-record recipe) | 153.92 / 153.66 / 153.16 (153.58) | 207.26 / 201.76 / 203.55 |
| **int8kv, SHRINK=0 (v1 recipe)** | **157.63 / 157.87 (157.75): +2.7% vs record, +4.6% vs fp32kv SHRINK=0** | 201.08 / 204.26 |
| fp32kv, SHRINK=0 | 150.67 / 150.88 (150.78) | 202.09 / 203.36 |
| bf16kv production | 163.06 / 163.58 (163.32) | 202.29 / 204.08 |

Hygiene and excluded rows:
- One cap abort: row 5 int8 ctx8192 prefill hit 0.412 s at 13:13:25Z. Not p1E's run; the display/kwin and a hermes process were active
  on dusk. That row is void.
- Row 8's short row overlaps p1E's 512-wrap emits (13:19:31–13:20:09Z) and is void.
- Host MemAvailable was 3.4–7.7 GiB, much lower than earlier days.
- The short rows are noisy (±2%, a busier host), so no short-row conclusion.

Reading: int8 recovers ~5% at depth over fp32 KV at equal SHRINK, but stays ~3.5% under bf16. The LOP path pays 2×LD per-element scale
multiplies. Folding the K scale after the reduction and the V scale into the weight (as the split paths already do) is the obvious v1.1.
Full-config numbers (gate + speed) need an owner window.
