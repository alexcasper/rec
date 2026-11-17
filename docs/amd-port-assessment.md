# AMD Port Assessment (bead rec-867, P1)

Survey of vendor-specific paths in rec and a CPU-only baseline run on dawn
(Ryzen 7 8700G, 16 cores, 46 GiB RAM, Linux 6.18, WSL2). Scope: cheap + local;
no GPU bring-up. Date: 2026-11-17.

## Headline finding

rec is **not** a CUDA engine. It is an **Apple-Silicon MLX + Metal** engine
("Flint inference engine for Apple Silicon"). The vendor coupling to abstract
is **MLX + Metal shaders**, not CUDA/NVIDIA. That changes the port target:

- There is **no MLX backend for ROCm** (as of mlx 0.32.0 — upstream ships
  macOS/Metal, Linux CPU, and Linux CUDA wheels). A ROCm port therefore means
  either a **torch/ROCm rewrite** of the model layer plus a **HIP/Triton
  rewrite** of the Metal megakernel, or a **Vulkan compute** rewrite.
- The decode fast path is one **custom Metal compute megakernel** (~1,300 lines
  of MSL embedded as Python strings, specialized per Apple chip at load time).
  This is the single largest port item and has no drop-in ROCm equivalent.
- A **portable pure-MLX core** exists (quantized prefill + decode, INT4 affine
  weights, INT8 embeddings, bf16 KV) and **runs on dawn's CPU today** with the
  pinned versions — see baseline below. It is a debug/smoke-test path only.

## Vendor-specific inventory

| # | Location | Vendor coupling | What it does | AMD translation | Cost |
|---|----------|-----------------|--------------|-----------------|------|
| 1 | `engine/pyproject.toml` | `mlx==0.32.0` (Metal-only on macOS; Linux wheels are CPU or CUDA) | Everything | No ROCm backend exists → torch-ROCm or Vulkan rewrite of the ops layer | **Blocking decision** |
| 2 | `engine/src/rec_mlx/engine.py:56-58` | `platform.system() != "Darwin" or machine != "arm64"` gate + `sysctl hw.memsize` + 48 GiB min-RAM | Hard refuse on non-Mac | Replace with platform probe (ROCm visible / Vulkan / CPU) and generic memory check | Trivial |
| 3 | `engine/src/rec_mlx/engine.py:154,182` | `mx.new_stream(mx.gpu)` | Metal streams, thread-local | `mx.gpu` maps to CUDA/ROCm device in MLX-CUDA; CPU on cpu backend | Easy |
| 4 | `engine/src/rec_mlx/kernels/device.py` | Apple chip table (`applegpu_g16s/g17s`, M4/M5) raises on unknown device | Resident-kernel tuning profiles (workers, prefill chunk) | Add RDNA profiles: dawn 8700G iGPU = `gfx1103` (780M); dusk GPU = TBD (`rocm-smi`) | Easy |
| 5 | `engine/src/rec_mlx/kernels/megakernel.py` (~1,300 lines) | MSL (Metal Shading Language) decode megakernel: RMSNorm/simd_sum, RoPE, attention, MoE dispatch, quantized matmuls, sampling — all in one launch; `FullModel.__init__` asserts `applegpu_*` arch + fixed Flint shapes (49 layers, hidden 2048, vocab 150144) | Whole-model decode in a single fused kernel | HIP rewrite (wave32 ↔ simd-group ops map well: RDNA wavefront = 32 lanes like Apple simdgroups) or Triton-on-ROCm; transforms (`expert_teams`, `routing_attention`, `mixed_bits`, `activation_optimization`) rewrite too | **Very high** |
| 6 | `megakernel.py:978-980` `_metal()` → `mx.fast.metal_kernel` | Metal custom-kernel launcher | Compiles + launches MSL | No ROCm equivalent in MLX; needs `mx.fast.cuda_kernel`-style path (CUDA backend only) or out-of-MLX HIP launcher | High |
| 7 | `kernels/m4_max.py`, `kernels/m4_max_prefill.py`, `profile_for_device` M4/M5 branches | Per-chip kernel specialization | Chip-tuned prefill/decode variants | Dead code on AMD until profiles exist; the transform pipeline itself is text-editing of MSL — retarget to HIP sources | High (part of #5) |
| 8 | `engine/native/grammar_native.cpp` + `native/build.py` | nanobind C++ against `libmlx`; build hardcodes `clang++`, `-undefined dynamic_lookup`, `@loader_path` rpath (macOS-only flags) | xgrammar grammar-mask plumbing into the decode kernel | C++ itself portable; fix build.py for Linux compilers/link flags | Easy-medium |
| 9 | `kernels/masked_sampling.py:102-130` | grammar_native handoff into Metal decode | Constrained sampling on GPU | Port with #5, or fall back to CPU-side masking | Medium |
| 10 | `model.py` `MixedQuantizedKVCache` (`mx.quantize`/`mx.quantized_matmul`, `mx.fast.scaled_dot_product_attention(..., sinks=...)`) | MLX ops (backend-neutral) | Quantized KV + attention with attention sinks | Portable **if** a backend exists; verified working on CPU | Portable ✓ |
| 11 | `cli/platform.go:21-45` | `GOOS != darwin || GOARCH != arm64` + `sysctl machdep.cpu.brand_string` + `checkAppleChip` | Go CLI install/preflight gate | Add linux/amd64 path with ROCm/Vulkan detection | Easy |
| 12 | `cli/manifest.go:84,152` + `packaging/flint-int4-mlx.json` | Manifest pins `os: darwin, arch: arm64, backend: metal`; Go CLI refuses mismatches; embeds release manifest, downloads macOS runtime server binary | Distribution/validation | New manifest variant (linux/amd64/rocm) + rebuild runtime binary | Medium |
| 13 | `engine/src/rec_mlx/{server,chat,prefix,scheduler,cache,checkpoint,grammar}.py` | Pure Python / stdlib / xgrammar | Serving, prompt cache, prefix scheduling, SHA-256 checkpoint validation | Portable as-is | Portable ✓ |

Checkpoint facts: real Flint weights are INT4-affine (group 64) + INT8
embeddings + bf16 KV, **21.1 GiB** across 5 shards, published at
`huggingface.co/RecursiveSI/flint-int4`; manifest (`packaging/flint-int4-mlx.json`)
requires `min_ram_bytes = 48 GiB` and context 262,144 (vs incumbent llama.cpp
flint at 131k). Weight format itself is vendor-neutral (safetensors) — no
port needed for the data, only for the code that consumes it.

## CPU baseline on dawn (mlx[cpu]==0.32.0, pinned deps)

Setup that worked (repo untouched; venv at `engine/.venv`, gitignored):

```bash
cd engine
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "mlx[cpu]==0.32.0" \
    "mlx-lm==0.31.3" "transformers==5.16.1" "xgrammar==0.2.1"
# NOTE: plain `mlx==0.32.0` is NOT installable on Linux — import fails with
# `ImportError: libmlx.so` unless the [cpu] (or [cuda]) extra is used.
```

The `Engine` class cannot start on Linux unmodified (gate #2), so the baseline
drives the portable core directly through the production path: production
`checkpoint.validate()` → `QATModel` (nn.quantize INT4/INT8 + strict safetensors
load) → `native` model prefill/decode with mlx-lm caches. Scripts:
`engine/.build/cpu_baseline.py` (functional) and `cpu_baseline_mid.py`
(Flint-shaped). Synthetic checkpoints under `engine/output/` (gitignored).

| Run | Model shape | Weights | Result (dawn CPU, 16 threads) |
|-----|-------------|---------|-------------------------------|
| Functional (tiny) | 2L, hidden 128, 8 experts, vocab 512 | ~1 MiB | prefill ~3,000 tok/s; decode **1,478 tok/s** — path proven end-to-end |
| Flint-shaped (mid) | 8L, hidden 2048, 32 experts top-2, vocab 150144, head_dim 128, sliding 513 | 1.03 GiB INT4/INT8 | prefill 512 tok: **2.7 tok/s**; decode: **2.8 tok/s** (greedy, single stream, bf16 KV); peak 2.46 GiB |

Capability probes on `mlx[cpu]` 0.32.0: `mx.compile` ✓;
`mx.fast.scaled_dot_product_attention(..., sinks=, mask=)` ✓; `mx.quantize`/
`quantized_matmul` ✓ (KV quantization needs head_dim divisible by group 64 —
Flint's 128 ✓).

Extrapolation: real Flint is ~20× this model's weights and 49 layers → well
under 1 tok/s on CPU. **CPU is a correctness/smoke-test lane, not a serving
lane.** The Vulkan llama.cpp incumbent on dusk (131k ctx) is orders of
magnitude ahead; only ROCm/HIP or Vulkan compute on the GPU can compete.

Full-CPU Flint run (21.1 GiB weights) is possible on paper (46 GiB RAM) but
does not fit the 48-GiB check and is not worth it pre-port; defer.

## Bugs / blockers found during the baseline (all pre-existing)

1. `model.py:82`: `MixedQuantizedKVCache.update_and_fetch` calls
   `self.keys_and_values()`, which does not exist on mlx-lm 0.31.3's
   `QuantizedKVCache` → `AttributeError` on the first full-attention decode
   step via `Model.make_cache()`. Latent: the Metal engine uses its own
   `FixedKVCache`; anything relying on `make_cache()` crashes.
2. Linux `mlx` wheel is broken without the `[cpu]`/`[cuda]` extra (see above).
3. `transformers==5.16.1` `save_pretrained` did not emit
   `special_tokens_map.json` in this run, while `checkpoint.validate()`
   requires it — affects checkpoint export tooling, not serving.
4. `native/build.py` uses macOS-only linker flags and `clang++` — wheel build
   fails on Linux with a stock toolchain (dawn currently has no system C++
   compiler; `nix shell nixpkgs#clang` is the workaround).

## Recommended port order

1. **Linux/CPU lift (cheap, days).** Remove the darwin/arm64 gates (items 2,
   11, 12), fix `build.py` linker flags, add `mlx[cpu]` handling, fix the
   `keys_and_values` bug. Deliverable: engine boots and serves on dawn CPU;
   CPU smoke-test in CI. No GPU needed.
2. **ROCm decision spike on dusk (P2).** Confirm dusk's GPU (`rocm-smi`),
   install ROCm + PyTorch ROCm wheel, and rewrite `model.py` (360 lines, the
   only nontrivial portable model) to torch as a spike. Measure tok/s of the
   *unfused* torch path on the real Flint weights — that is the floor a
   megakernel port must beat, and possibly competitive already for prefill.
3. **Decode fast path (the big one).** Port the megakernel to HIP (RDNA
   wave32 maps well onto the MSL simd-group code) or Triton-on-ROCm; port the
   device-profile table (#4) and the grammar-mask handoff (#9). This is where
   rec's speed advantage lives; without it, a torch port is just another
   llama.cpp-adjacent server.
4. **Vulkan fallback / dawn iGPU (P3).** dawn's 780M (gfx1103) iGPU has
   spotty official ROCm support (HSA override needed); Vulkan compute would
   duplicate llama.cpp's niche. Recommendation: don't port Vulkan — instead
   use dawn as an ROCm-experiment smoke target, and treat Vulkan as incumbent-
   only.

## Risks

- **Megakernel equivalence:** the MSL is tuned per-chip by string transforms;
  behavioral parity (bit-exactness of sampling paths) is asserted by tests
  that currently can only run on Apple hardware.
- **Fixed-shape asserts:** `FullModel` hardcodes Flint's exact architecture —
  any AMD path can reuse that, but generalizing later means re-touching it.
- **Upstream drift:** upstream is macOS-focused; keep the Linux lift as
  upstreamable PRs (per repo convention), keep the Metal megakernel untouched.
- **Distribution:** the Go CLI ships a prebuilt macOS runtime binary and an
  embedded manifest; a Linux lane needs its own build+sign pipeline
  (`packaging/`), not just code changes.
- **Memory gate:** 48-GiB min-RAM is tied to Metal unified memory assumptions;
  on dawn (46 GiB) even a healthy port must re-derive the KV budget (real
  KV block 1.78 GiB + 21 GiB weights).
