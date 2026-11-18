# ROCm model layer — tensor layout contract (rec-9na)

Owner: Claude arm (correctness + model layer). Consumers: rec-tan (HIP
megakernel), any future fused dequant-matmul (rec-aip). This file is the
binding contract. If a kernel needs a different layout, change this file in
the same PR that changes the kernel.

Ground truth for semantics is `engine/src/rec_mlx/model.py` (the MLX model).
The torch reference is `engine/src/rec_mlx/torch_rocm.py`, wired into the
engine as backend `rocm` via `engine/src/rec_mlx/torch_runtime.py`.

## 1. Checkpoint format (what is on disk)

I verified this against the production export
`RecursiveSI/flint-int4@3b93f905` by reading the safetensors header of
`model-00000.safetensors` over HTTP range requests (2026-11-18):

| Tensor | dtype | shape | notes |
|---|---|---|---|
| `*.{q,k,v,o}_proj.weight`, MLP/expert `*.weight` | **U8** | `[..., out, K/2]` | INT4 affine, 2 nibbles/byte, **low nibble = even element** |
| `*.scales`, `*.biases` | **BF16** | `[..., out, K/64]` | group 64 along K; `w = q*scale + bias` |
| `*.switch_mlp.*` | U8 / BF16 | `[E, out, ...]` | experts stacked on axis 0 (E=256 in Flint) |
| `model.embed_tokens.weight` | U8 (inferred) | `[V, H]` | **INT8** affine (`embedding_bits: 8`), group 64. It lives in a later shard whose header I did not read, so U8 is inferred from the U8 convention; the loader accepts U8 or U32 |
| norms, `q_norm`/`k_norm`, `softmax_offset` | BF16 | `[H]`, `[head_dim]`, `[heads]` | |
| `*.mlp.gate.weight` (router) | BF16 | `[E, H]` | **not quantized** |
| `*.mlp.gate.expert_bias` | — | — | **absent** (`router_expert_bias: false`); loaders fill zeros |

Synthetic checkpoints (`python -m rec_mlx.synthetic`) use U32 packed words
and F32 floats instead. The bytes are equivalent: U32 word `j` holds elements
`8j..8j+7` at bits `4*(i%8)`, which is the same bytes as the U8 layout on
little-endian. Loaders must accept both. `tests/test_torch_rocm.py::
test_production_format_checkpoint_loads_and_matches` covers the U8/BF16/no-bias
form.

## 2. Device-resident weights (decision)

- **No repacking.** Upload the checkpoint bytes as they are. Packed weights
  are viewed as `int32 [..., out, K/8]` in torch, or `uint8*` / `uint16*` in
  HIP. Scales and biases keep the **checkpoint dtype** (bf16 in production),
  `[..., out, K/64]`, row-major. Dequant math is fp32: `q*float(s) +
  float(b)`.
- **Expert strides.** Expert `e` of a stacked tensor `[E, out, K]` starts at:
  - packed: `e * out * K/2` **bytes** (`uint8*`)
  - scales/biases: `e * out * K/64` **elements** of a `bf16*`, which is
    `e * out * K/64 * 2` bytes.

  Never add a byte stride to a typed pointer. That exact bug was rec-hjr: a
  later expert read the next tensor's bytes, giving NaN logits.
- **INT4 pre-scale trick (MSL qdot).** Each `uint16` covers 4 consecutive
  elements. The MSL kernel pre-divides `x` by `1, 16, 256, 4096` and masks the
  nibbles in place. The period is **4 per uint16**, i.e. element `j`
  gets divisor `16^(j % 4)`, not `16^j`. That was rec-qib: K=256 rows dropped
  elements 4..7 of each lane.
- **Bias fold.** `dot(w, x) = scale * Σ q·x + bias * Σ x` per group. Kernels
  may use this, but the group sum `Σ x` must cover all 64 elements of the
  group.
- Dense params (norm weights, sinks, router weight) may be held as fp32 on
  device. They are tiny.

## 3. Activations and numerics (decision)

- Residual stream and inter-op activations are **bf16**. Matmul accumulates
  in fp32. RMSNorm, softmax, sigmoid, SiLU and RoPE angles are computed in
  fp32 and rounded to bf16 on store.
- Block wiring (from `model.py`): `h = x + attn(rmsnorm(x))`, then
  `out = h + mlp(rmsnorm_post(h))`. The residual is never normalised.
- Attention: q/k get a per-head **RMSNorm** (`q_norm`/`k_norm` weights), then
  NeoX half-split RoPE (`theta=10000`, `inv_freq = theta^(-2i/head_dim)`) on
  rope layers only (`no_rope_layer_indices` = the full-attention layers).
  Scale is `1/sqrt(head_dim)`. **Sinks:** `softmax_offset[h]` is appended as
  one extra logit column, softmaxed with the scores, then dropped.
- Sliding layers attend to keys with `q_pos - k_pos <= 512` (window 513
  including the current token).
- **MoE router:** `scores = sigmoid(x @ Wr^T)` in fp32. Select top-k on
  `scores + expert_bias`. The bias steers **selection only**. Mixing weights
  are the **unbiased** `scores[top]`, renormalised to sum to 1 (not softmax).
  The output adds the shared expert unconditionally. Expert MLP is
  `down(silu(gate(x)) * up(x))`. Do not drop the SiLU (rec-qib).
- Logits are returned to the engine as fp32.

## 4. KV cache (decision)

The KV cache is **bf16**, stored after k_norm and RoPE (rotated keys), with
values raw. The kernel contract **matches the Metal production layout**
(`rec_mlx/cache.py`; verified from the `flint_write_kv` addressing and
`KVStorage.layer()`), so MSL→HIP translation stays 1:1:

- **Full-attention layers:** per layer, one region `[2 (K|V)][kv_heads][cap][head_dim]`,
  head-major. Element `(kv, h, pos, d)` sits at
  `base + ((kv*kv_heads + h)*cap + pos)*head_dim + d`. Positions are linear;
  `cap` grows in blocks of 65536 tokens (`BLOCK_TOKENS`). A head's keys are
  contiguous over positions, so decode attention streams them.
- **Sliding layers:** a ring of 513 positions per head
  (`FixedSlidingKVCache`: K and V `[1, kv_heads, 513, head_dim]`),
  `slot = pos % 513`.
- The torch reference (`torch_rocm.Model`) uses the same head-major order,
  `(k, v)` each `[B, kv_heads, T, head_dim]`. It applies the sliding window
  by mask, not by ring: semantically identical, but memory grows past 513 on
  sliding layers. That's fine for the reference path; a ring is a perf
  follow-up.
- The tiny validation megakernel uses `[layer][kv_head][K|V][t][head_dim]`,
  which nests K|V *inside* the head. That is fine for the tiny phase. The
  mid/full kernel should move to `[K|V][kv_head][t][head_dim]` per layer
  (rec-tan).

## 5. Validation hooks

| What | Where | Runs on |
|---|---|---|
| torch layer == MLX model (dequant bit-exact, fp32 logits, window > 513, KV incremental, prod-format ckpt, engine `rocm` path) | `engine/tests/test_torch_rocm.py` | any Linux (CPU torch) |
| GPU vs CPU conformance of the torch layer | `engine/tests/torch_rocm_tiny_check.py --ckpt DIR --device cuda` | ROCm host |
| HIP tiny megakernel == MLX model on the exact bank | `native/megakernel_hip/tiny_mk_driver.py` → `tiny_mk_np_reference.py` (any host) / `tiny_mk_mlx_reference.py` (ground truth) | ROCm host |
| numpy reference pinned to MLX | `engine/tests/test_tiny_mk_reference.py` | any |
| degenerate MoE router rejected at load | `rec_mlx.checkpoint.assert_router_nondegenerate` (QATModel + torch loader) | any |

Synthetic test data must be **non-degenerate**: zero-centred weights and
varied greedy tokens. An all-positive bank makes one token repeat forever and
hides wiring bugs (rec-qib). A zero router makes every expert tie (rec-9as).
