# rec-xh7 decision matrix — full-Flint megakernel vs incumbent (SKELETON)

Status: DRAFT skeleton, 2026-11-20 (rec-xh7 parent holder: p1E lane per
Bedivere coordination). Every number cites its source comment/commit; rows
carry cotenant state per the hygiene protocol — never compare across
different cotenant states. The decision this matrix feeds: does the
megakernel path beat the llama.cpp-Vulkan incumbent on the real Flint
config, and at what quant/bytes trade?

## Ruling in force (step-0a executive rec, Alex to veto)

Consume `RecursiveSI/flint-int4@3b93f905` (mlx-affine INT4/BF16-s-b, INT8
embed) directly in the megakernel — no requant, no in-kernel i-quant
dequant. bpw-matched incumbent row = same source quantized to Q4_K_M GGUF
with llama.cpp's offline quantize tool; UD-Q3_K_XL row stays as the
as-shipped quality anchor.

## Config under test

Full Flint (config.json @ 3b93f905): 49L, H=2048, 32Q/4KV, head_dim 128,
vocab 150144, **256 experts top-6**, moe_i 512, shared 1024, dense [0,1],
sliding 513. Attention/MLP geometry is IDENTICAL to the validated mid-8L;
deltas = depth (2→49), experts (8→256), top-k (2→6).

## Rows (filled)

| # | arm | config | decode tok/s | ctx depth | cotenants | provenance |
|---|---|---|---|---|---|---|
| R1 | mid-8L persistent megakernel (INT4 affine, fused q4_gemv) | mid-8L synthetic, 0.944 GiB bank | **120.78** decode32 (ABBA differential, workers=16; pair range 117.08–123.49) | after 512 | 3 llama-servers idle | rec-aip 2026-11-19 07:58Z, d1f6208; 6/6 rec-pp2 PASS; raw json committed |
| R2 | llama.cpp-Vulkan, mid-8L GGUF **UD mix (as-shipped quant mix)** | mid-8L | **120.1** | 512 / **108.1** @ 8192 | idle | rec-xh7 step 0, 2ea0c9f, docs/benchmarks/xh7-step0/README.md |
| R3 | llama.cpp-Vulkan, mid-8L UD mix + **q4_0 lm_head** | mid-8L | **161.5** | 512 / **139.3** @ 8192 | idle | same as R2 |
| R4 | llama.cpp-Vulkan, mid-8L **pure q4_0** (megakernel's bytes/token) | mid-8L | **201.9** | 512 / **168.6** @ 8192 | idle | same as R2 |
| R5 | llama.cpp-Vulkan, **full Flint via :8080** server | full 49L | **26.6** (qwen stopped) / 26.8 (qwen up) | 8k | mixed (recorded per run) | rec-xh7 step 0, 2ea0c9f |
| R6 | engine e2e, rocm backend, **real mid-8L weights bf16** (rocm-bf16) | real mid-8L | **7.22** | greedy gen, torch-native sampling | 3 llama-servers UP | rec-9na comment 2026-11-20 (torch-fp32 gate; fp32 row: 5.94/808.1) |
| R7 | engine e2e rocm-fp32 | real mid-8L | **5.94** / prefill 808.1 | same as R6 | same | same |
| R8 | **full-config megakernel** (49L / 256E / top-6, bank `~/xh7/flint-int4.bank` 21.019 GiB, bit-exact verified) | **full Flint INT4-affine** (~4.5 bpw-equiv) | **35.45** decode32 (w64; w32 34.91, w16 31.43; ABBA pairs valid) | after 512; short-ctx 54.1/67.6/69.9; e2e-516 39.5–47.1 pos/s | recorded per log line | **HOST: albion R9700 (discrete 32 GB)** — rec-u9q GREEN, e3c5ed6 on rec/gemini-kernels, docs/benchmarks/u9q-albion-2026-11-20/; sign-off: fp32-KV 3/3 under adopted spec (a), step_rel ≤2.3e-6; bf16-KV 0/3 = reference-sensitivity artifact (ref itself moves 5.5e-2–8.5e-2 fp32→bf16 KV; router gaps to 3e-7 cascade over 49 layers) — **judged by token agreement**, no logit-rel gate. Scaling context: same-card mid-8L 196–261 tok/s |
| R9 | **llama.cpp-HIP build 513ad7ae** (ROCm/HIP backend, NOT Vulkan; -ngl 99 -fa on, 8 threads), **UD-Q3_K_XL as-shipped** (~3.5–3.9 bpw mix) | **full Flint** | **76.3 ± 3.2** decode32@512 (5 reps 70.9–79.1); prefill512 **4162.8 ± 143.5** (3 reps) | 512 | **none** | rec-b3u 17:27Z, albion:~/models/u9q_inc_mix_{d512,p512}.jsonl (+ .err, u9q_incumbent.sh) |
| R10 | **llama.cpp-HIP 513ad7ae/2660, pure Q4_0** — 21424 MiB GGUF = **4.52 BPW, byte-matched to affine INT4 g64+bf16** (converter validated: mid f16 GGUF = dusk step-0 GGUF on all 127 tensors, rel ≤1e-3) | **full Flint** | **82.78 ± 5.34** decode32@512 (5 reps 76.5–87.3); prefill512 **4517.12 ± 34.07** (3 reps) | 512 | **none** | rec-b3u, commit 6866868 on rec/gemini-kernels, docs/benchmarks/b3u-q40-albion-2026-11-20/ (q40_{d512,p512}.jsonl, q40_rows.log with busy+cotenant per line) |

Cross-reads (with caveats): R1 vs R2 = **parity on the shipped mix**
(1.006×); R1 vs R4 = **0.60× at equal bytes** (~38 vs ~64 GB/s effective)
— the gap fused-dequant depth must close. R6's do-not-compare note: the
stale rec-rfp 2463 prefill / 37.2 decode rows are superseded.
**HOST WARNING: R8 (albion R9700 discrete) must NOT be compared against
R5 (dusk 890M UMA) — different silicon and memory path.**

## Head-to-head, same card (albion R9700), byte parity — DECISION READ

**R8 megakernel 35.45 vs R10 incumbent 82.78 → incumbent leads 2.33× at
byte parity. The decision read HOLDS AGAINST the megakernel at full
config.** The deficit is not quant choice (R10 is byte-matched to the
megakernel's weights) — it is the megakernel's fused-dequant effective
bandwidth (~38 vs ~64 GB/s, per the R1-vs-R4 cross-read), which is the
next optimization lever for the kernels lane. Consequence: the Alex
step-0 layout decision (requant-to-affine vs in-kernel i-quant dequant)
**no longer changes the full-config read** — either path lands the
megakernel behind llama.cpp at byte parity until dequant bandwidth is
fixed. Secondary caveat on R10 itself: pure-Q4_0 requant fidelity vs
torch fp32 CPU is 11/16 + 11/16 teacher-forced (flips at logit margins
up to 4.6) — a quality cost of the byte-matched quant, present for both
arms' quality story and worth a quality row if this goes external.

## Rows (open)

| # | arm | config | owner | status |
|---|---|---|---|---|
| O1 | ~~full-config megakernel~~ → **FILLED as R8** | full Flint INT4-affine | w11:p1F | rec-u9q closed GREEN (e3c5ed6) |
| O2 | ~~pure-q4_0 byte-fair incumbent~~ → **FILLED as R10** | full Flint | w11:p1F | 6866868 (verified) |
| O3 | quality row: ppl/agreement affine-INT4 vs UD-Q3_K_XL vs unquantized | full Flint | — | MOOTED pending Alex (note: R10's own fidelity 11/16 makes a quality row more relevant if this goes external) |
| O4 | ~~llama.cpp incumbent on albion R9700~~ → **FILLED as R9 + R10** | full Flint | w11:p1F | rec-b3u remaining: origin push of e3c5ed6/6866868 |

**No open matrix rows.** The decision read holds against the megakernel
at byte parity; the follow-on work item is the kernels-lane fused-dequant
bandwidth optimization (~38 → ~64 GB/s effective), which is a new bead,
not a matrix hole. Backend framing note: R9/R10 are llama.cpp-**HIP**
(ROCm); R2–R5 were recorded as Vulkan (step-0) — read the mission's
"Vulkan incumbent" as "llama.cpp (Vulkan on dusk, HIP on albion)".

## Provenance / hygiene notes

- Bedivere-cited incumbent "31.1 decode @ 8k ctx solo (~10.2 with
  cotenants)" was **not reproduced** through the :8080 server (R5:
  26.6–26.8); its provenance is not in the repo. R5 is the recorded
  number; reconcile before any external quote.
- The old eager-torch INT4 rows (2.25 → 3.69–3.71) and the 37.2
  dense-bf16 floor are superseded/context-only (7fc236a); not ladder rows.
- All llama.cpp GGUF rows measured with: gated busy<5, ABBA order, 32
  clean runs, depth 512/8192, cotenant state recorded (step-0 README).
- Hygiene protocol for new windows: pause llama-qwen :8091 (ours), :8090
  needs E4B owner window, flint :8080 stays up, cotenant state per log
  line, busy<5 + ABBA.
