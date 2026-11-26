# rec-ywy full-config int8kv landing: GQA read path, sliding rings, determinism pin, gates, bench row (2026-11-25/26)

**Recipe pin (rec-czs):** full lib `/home/dusk/rec-ve/i8/fulls0/flint_mid.so` (sha256 `be2fa140e6060c07`), built in 3.9 s from HEAD
source `16e3dd85` (= 04e9d88 rec-ywy v1.1) with `-DFLINT_CFG_FULL=1 -DGEPA_ATTN_SPLIT=1 -DGEPA_ATTN_LOP=8 -DGEPA_SHRINK=0`
— the rec-ywy mid recipe (SHRINK=0) carried to full config. All five kernels present incl `mid_generate_fp32_int8kv`
(VGPR 250 / scratch 32 B / LDS 25272). **No kernel source change was needed**: the KV layout is per-kv-head
(MID_NKV=4) with sliding 513-slot rings at both configs; the int8 read paths (1-warp sinks / split non-LOP / split LOP)
already apply per-(kv-head, token) scales. The compile-time KV8 rejection (`flint_mid_megakernel.hip:1299`) covers the
GEPA_ATTN_GQA depth path + LDS + LOP_PF — none used by this recipe — and stays in place deliberately (rec-776 not needed).

## Hard rule — fp32kv builds byte-identical

| emit | md5 | baseline | verdict |
|---|---|---|---|
| wrap512 fp32kv (new int8 lib) | `dabb2919bbc8` | rec-3cr/rec-04i exact | **MATCH** |
| wrap512 bf16kv (new int8 lib) | `eec63518fa18` | rec-04i exact | **MATCH** |
| depth8192 fp32kv (new int8 lib) | `c142fc5397fb` | rec04i 3x-fresh exact | **MATCH** |

## Determinism (rec-czs) — PASS

depth8192 int8kv fresh sessions: **5/5 identical md5 `b18fb5eb7aae`** across three windows (W1 x3 flint-down 02:32–02:45Z,
W2 x1 02:53Z, W3 x1 04:20Z incl the pairs-mode session). `w1_det3.log` has the 3-repeat PASS line.

## Full-config gates (rec-3tv) — PASS

Scorer `score_i8_full.json` (dawn CPU torch reference, full flint-int4, eval_set_full_v1 sha 1b06d928, 8 items x 191 pos;
kernel ppl emits from W2 `--ppl-set`, 02:59–03:02Z):

| row | ppl_rel | top-1 agree |
|---|---|---|
| kernel fp32kv vs torch fp32 | 1.2e-7 | 100% |
| kernel int8kv vs torch int8 (impl check) | +0.146% | 98.76% |
| **gate int8kv vs fp32 ref** | **+0.776% (<1%)** | **97.58% (>=97%)** → **PASS** |
| torch int8 vs torch fp32 (baseline) | +0.63% | 97.71% |

Absolute ppl: fp32kv 3.2803 vs int8kv 3.3058. Caveat: impl check 98.76% is below the 99% aspiration (rounding-boundary
flips amplified by depth) but above the binding 97% tier; the binding quality gate passes.

## Bench row (rec-7oc) — W3 quiet window, 2026-11-25 04:16–04:30Z

| KV mode (same lib, SHRINK=0) | decode tok/s @8192 (ABBA 12/12, depth 8215..8265, k=3) |
|---|---|
| **int8kv** | **32.55** |
| bf16kv | 32.62 |
| fp32kv | 30.77 |

vs incumbent llama.cpp Vulkan 26.80: **+21.4%**. vs same-lib fp32kv: **+5.8%**. Parity with bf16kv.
The promoted 33.40 rec-04i row is NOT same-protocol comparable (bf16-KV mode, SHRINK=1+fence codegen; the SHRINK=0
codegen family measured 32.73 in that mode — this window's bf16kv 32.62 is consistent). Depth path is not
KV-bandwidth-bound at 8k (tA 61 ms fixed launch cost dominates), which is why halving KV bytes vs bf16 does not add speed.

## VRAM (measured probe, 2026-11-26 00:40Z) — v1 recipe does NOT save VRAM

Session footprint incl identical 21.0 GiB bank at 8192 cap: fp32kv 22360 MiB, bf16kv 22352 MiB, **int8kv 22486 MiB
(+126 over fp32kv)** — the lazy int8+scales mirror is additive on top of the base buffers (consistent with rec-ywy
"kv8 buffers keep fp32/bf16 sessions' VRAM unchanged"). The expected 543→136–144 MB saving is not realized in v1;
int8kv is a speed/quality play, not a VRAM play. `vram_probe.{py,json}`.

## Windows / cotenants provenance

- W1 + W2 + W3 and the VRAM probe each stopped `llama-flint.service` per the rec-kcb :8080-down protocol and
  trap-restored it (health-verified 200 after every window; ops_flint seat was dead — w17:pQ operated the unit,
  one external restart observed 02:57:57Z Nov 25, restorer unidentified, no cron/timer found).
- W2's ABBA pairs aborted (dawn fleet memory-extraction churn kept the shared iGPU at 97–100% busy, `e4b:busy`);
  W3 armed a quiet-box wait (3x1s busy sum <15, 2 consecutive probes) and fired when the box went idle 04:16Z.
- Launch cap held everywhere (longest 0.188 s).

## Artifacts

`w1_{window,det3}.log`, `w2_{window,bench_i8}.log`, `window.log`+`bench_{i8,b16,f32}.log` (W3, npz dirs),
`ppl__{fp32kv,int8kv}.npz` (kernel side), `score_i8_full.json` (torch scoring), `vram_probe.{py,json}`.
Dusk originals: `~/rec-ve/i8/{fulls0,w1-20261125T023227Z,w2-20261125T024829Z,w3-20261125T033308Z}`.
