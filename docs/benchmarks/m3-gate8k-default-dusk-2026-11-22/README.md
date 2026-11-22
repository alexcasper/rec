# Discriminator window — DEFAULT full build (split-K off, GEPA_SHRINK=0) at depth 8192, dusk 2026-11-22 19:59–20:31Z

Coordinator-dispatched. :8080 DOWN 19:59:11Z → script 19:59:17–20:30:39Z (31.4 min, hard stop 42; all steps rc 0; longest launch
0.247 s, cap 0.25 s) → health 200 20:33:01Z; ops_flint in-band 29.48 tok/s @512 (~31.7 min down). Cotenants per line (`window.log`):
E4B 200, qwen3vl 200/200 (loopback), llama-qwen inactive, display ACTIVE, MemAvailable 16.8–17.2 GiB, VRAM free 38.7 GiB.
Lib `flint_full_def_288f4fa.so` f217f678…. Tiers: rec-kcb v1 (`docs/gates/rec-kcb-full-config-tiers-v1.md`, interim).

## Verdict: the depth-8192 fp32-KV failure is a BASELINE full-config property, not split-K merge error
| depth 8192, fp32kv vs torch fp32 CPU | last | step max | routing devs | tokens |
|---|---|---|---|---|
| default (split off), this window | **7.410e-2** | 0.2969 | 1 | exact |
| split-K + SHRINK=0 (18:03Z window) | 7.412e-2 | 0.2969 | 3 | exact |

Kernel vs kernel (`default_vs_split_kernel.txt`): identical tokens, last logits within 6.0e-5, steps within 1.7e-4; 65 positions
route differently, none before 3255 (split's relaxed merge order). The two kernels agree to ~1e-4 and both sit 7.4e-2 from the reference.
On mid at depth 8192 the same default build PASSES (8.1e-6). So the gap appears only at full config and depth: either a kernel
full-config path that both builds share, or the reference at full-config depth. Open: rec bead filed.

## Other checks
| check | result |
|---|---|
| determinism, default fp32kv @8192 | **PASS**: 2 fresh sessions byte-identical (md5 8e8c0124…; D1 and D2 are separate invocations) |
| production bf16kv @8192 emit | done (md5 f5d6b66e…), 127.0 ms/position at depth; bf16-KV tier is informational at full config, not scored |
| speed @8192 (12 runs requested) | **not measurable under the 0.25 s cap**: one single-position launch at depth takes 245 ms, so an ABBA B launch (≥2 positions) would break the cap. Bounds: ≥ 4.08 tok/s (1/tA); prefill rate at depth implies ~7.9 tok/s. Either way far below the incumbent's 26.80 and split-K's 21.55 |
| default prefill 8192 | 619 s fp32kv (118.5 ms/position at depth), against split's 325 s |

Implication for rec-e5d: split-K is worth 2.7× at depth over the default build (21.55 vs ≲8 tok/s) and is numerically equivalent to it
(~1e-4), but both builds share the depth correctness gap, and split-K still runs at 0.80× llama.cpp.
