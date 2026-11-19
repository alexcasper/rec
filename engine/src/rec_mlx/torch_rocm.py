"""Torch model layer for the ``rocm`` backend (rec-9na).

Mirrors the portable pure-MLX core (``rec_mlx.model``) on torch so it runs on
ROCm (or any torch device). Semantics follow the MLX checkpoint exactly:

- INT4 affine weights (MLX packing: group 64 along K, element j of a uint32
  word at bits [bits*j, bits*(j+1)), dequant = q*scale + bias). Weights stay
  packed in device memory and are dequantized inside the matmul.
- INT8 embeddings, dequantized per gathered row.
- bf16 KV cache (k is stored after q/k RMSNorm and NeoX RoPE); full layers
  grow, sliding layers use a 513-slot ring (slot = pos % window).
- Attention sinks (``softmax_offset``) appended as an extra softmax column.
- Sliding-window layers attend to positions within ``sliding_window - 1``.
- MoE: sigmoid router with additive expert bias for top-k selection, gate
  weights from the unbiased sigmoid, renormalised; plus a shared expert.

Torch only; no mlx import. Pure-torch dequant is the reference path. A fused
dequant-matmul kernel is the perf follow-up (rec-aip).
"""

from __future__ import annotations

import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import assert_router_nondegenerate

GROUP_SIZE = 64

# safetensors dtype -> numpy storage dtype. BF16 has no numpy dtype: it is read
# as raw uint16 bits and reinterpreted in torch (lossless). The production
# Flint export (RecursiveSI/flint-int4) is U8 packed weights + BF16
# scales/biases/norms/router; the synthetic checkpoints are U32 + F32.
_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U32": np.uint32,
              "I32": np.int32, "U8": np.uint8, "I8": np.int8, "U16": np.uint16}


@dataclass(frozen=True)
class StTensor:
    data: np.ndarray
    dtype: str  # safetensors dtype tag

    def torch(self, device=None, float_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Float tensors (F32/F16/BF16) -> torch, optionally cast to ``float_dtype``."""
        t = torch.from_numpy(self.data)
        if self.dtype == "BF16":
            t = t.view(torch.bfloat16)
        if float_dtype is not None:
            t = t.to(float_dtype)
        return t.to(device) if device is not None else t

    def float32(self) -> np.ndarray:
        if self.dtype == "BF16":
            return (self.data.astype(np.uint32) << 16).view(np.float32)
        return self.data.astype(np.float32)


def load_safetensors(path: Path) -> dict[str, StTensor]:
    """Read a .safetensors file with stdlib + numpy only."""
    out: dict[str, StTensor] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        header.pop("__metadata__", None)
        for name, info in header.items():
            dt = _ST_DTYPES.get(info["dtype"])
            if dt is None:
                raise ValueError(f"{name}: unsupported safetensors dtype {info['dtype']}")
            start, end = info["data_offsets"]
            f.seek(base + start)
            buf = f.read(end - start)
            out[name] = StTensor(np.frombuffer(buf, dtype=dt).reshape(info["shape"]).copy(), info["dtype"])
    return out


@dataclass(frozen=True)
class Config:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int
    eps: float
    theta: float
    sliding_window: int
    layer_types: tuple[str, ...]
    no_rope: frozenset[int]
    dense_layers: frozenset[int]
    num_experts: int
    topk: int

    @staticmethod
    def from_json(path: Path) -> "Config":
        c = json.loads(Path(path).read_text())
        return Config(
            hidden=c["hidden_size"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            layers=c["num_hidden_layers"],
            vocab=c["vocab_size"],
            eps=c["rms_norm_eps"],
            theta=c["rope_theta"],
            sliding_window=c["sliding_window"],
            layer_types=tuple(c["layer_types"]),
            no_rope=frozenset(c["no_rope_layer_indices"]),
            dense_layers=frozenset(c["mlp_only_layers"]),
            num_experts=c["num_experts"],
            topk=c["num_experts_per_tok"],
        )


def dequant(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
            bits: int, dtype: torch.dtype) -> torch.Tensor:
    """MLX affine dequant. ``packed`` is int32 [..., K*bits/32]; scales/biases
    are [..., K/GROUP_SIZE]. Returns [..., K] in ``dtype``.

    Bit-identical to the original formulation (int32 shift/mask unpack, scales
    and biases ``repeat_interleave``d to [..., K], fp32 ``q * s`` then ``+ b``
    with no FMA, then cast), but it reads the packed words as bytes and
    broadcasts s/b per group instead of materialising them (rec-aip profile s4).
    On the dusk 890M the mid lm_head dequant goes from 227 ms / 4.59 GiB
    transient to 134 ms / 1.72 GiB (engine/tests/test_torch_rocm_dequant.py
    pins bit-identity)."""
    per_word = 32 // bits
    K = packed.shape[-1] * per_word
    groups = K // GROUP_SIZE
    if scales.shape[-1] != groups or biases.shape[-1] != groups:
        raise ValueError(f"scales/biases {tuple(scales.shape)} do not match K={K} / {GROUP_SIZE}")
    if bits in (4, 8) and sys.byteorder == "little":
        # Little-endian int32 word w holds element w*per_word + j at bits [j*bits, (j+1)*bits),
        # i.e. byte j*bits//8 (low nibble first for 4-bit) -- the same order as the shift unpack.
        b = packed.contiguous().view(torch.uint8)
        q = torch.stack((b & 15, b >> 4), dim=-1) if bits == 4 else b
    else:
        shifts = torch.arange(per_word, device=packed.device, dtype=torch.int32) * bits
        q = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)  # mask kills sign bits
    q = q.reshape(*packed.shape[:-1], groups, GROUP_SIZE).to(torch.float32)
    q.mul_(scales.float().unsqueeze(-1)).add_(biases.float().unsqueeze(-1))
    return q.view(*packed.shape[:-1], K).to(dtype)


class QLinear:
    """Affine-quantized linear / switch-linear kept packed on device."""

    def __init__(self, packed, scales, biases, bits: int):
        self.packed = packed  # int32 [..., out, K*bits/32] (same bytes as the U8/U32 export)
        self.scales = scales  # checkpoint dtype (bf16 in production) [..., out, K/g]
        self.biases = biases
        self.bits = bits

    def dense(self, dtype: torch.dtype) -> torch.Tensor:
        return dequant(self.packed, self.scales, self.biases, self.bits, dtype)

    def expert(self, e: int, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize one expert of a stacked [E, out, ...] switch linear."""
        return dequant(self.packed[e], self.scales[e], self.biases[e], self.bits, dtype)

    def rows(self, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize only the rows named by ``ids`` (embedding lookup)."""
        return dequant(self.packed[ids], self.scales[ids], self.biases[ids], self.bits, dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def rope_neox(x: torch.Tensor, offset: int, theta: float) -> torch.Tensor:
    """x: [B, H, L, D]; NeoX (half-split) rotation, positions offset..offset+L-1."""
    _, _, L, D = x.shape
    half = D // 2
    pos = torch.arange(L, device=x.device, dtype=torch.float32) + offset
    inv = torch.pow(theta, -torch.arange(half, device=x.device, dtype=torch.float32) / half)
    freqs = torch.outer(pos, inv)
    c = freqs.cos()[None, None].to(x.dtype)
    s = freqs.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class SlidingKV:
    """Fixed 513-slot ring for one sliding-window layer (rec-1h9).

    Matches the layout contract (docs/rocm-model-layer.md s4) and production
    ``FixedSlidingKVCache``: K and V ``[B, kv_heads, W, head_dim]``, position
    ``p`` lives in slot ``p % W``. ``pos[slot]`` is the absolute position held
    (-1 = empty), so attention masks come from slot positions, not slot order.
    """

    def __init__(self, B: int, heads: int, window: int, head_dim: int, dtype, device):
        self.window = window
        self.k = torch.zeros(B, heads, window, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.pos = torch.full((window,), -1, dtype=torch.long, device=device)

    def write(self, k: torch.Tensor, v: torch.Tensor, qpos: torch.Tensor) -> None:
        """Store the newest min(L, W) positions of a chunk (older ones are out of window)."""
        n = min(k.shape[2], self.window)
        slots = qpos[-n:] % self.window
        self.k[:, :, slots] = k[:, :, -n:].to(self.k.dtype)
        self.v[:, :, slots] = v[:, :, -n:].to(self.v.dtype)
        self.pos[slots] = qpos[-n:]

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class KVCache:
    """Per-layer cache: growing ``(k, v)`` on full-attention layers, a
    :class:`SlidingKV` ring on sliding layers. ``offset`` = tokens consumed."""

    def __init__(self, cfg: Config):
        self.layers: list = [None] * cfg.layers
        self.offset = 0

    def __iter__(self):
        """(k, v) per populated layer (ring buffers for sliding layers)."""
        for entry in self.layers:
            if isinstance(entry, SlidingKV):
                yield entry.k, entry.v
            elif entry is not None:
                yield entry

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in self)


class Model:
    """Flint-shaped decoder on torch. Load with :func:`load_model`."""

    def __init__(self, cfg: Config, tensors: dict[str, torch.Tensor], qlin: dict[str, QLinear],
                 embed: QLinear, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.t = tensors  # dense params (norms, sinks, router, expert bias), already on device
        self.q = qlin
        self.embed = embed
        self.device = device
        self.dtype = dtype
        # Validation knob: round K/V to this dtype at the KV-cache storage points
        # (K after k_norm + RoPE, V after projection) while computing in `dtype`.
        # None = store in `dtype`. Used by engine/validation/megakernel_mid to
        # build an exact fp32-compute / bf16-KV reference (layout contract s4).
        self.kv_store_dtype: torch.dtype | None = None

    # -- helpers -----------------------------------------------------------
    def _lin(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.q[name].dense(self.dtype))

    def _mlp(self, x: torch.Tensor, tag: str) -> torch.Tensor:
        g = self._lin(f"{tag}.gate_proj", x)
        u = self._lin(f"{tag}.up_proj", x)
        return self._lin(f"{tag}.down_proj", F.silu(g) * u)

    def _attention(self, x, i, qpos, offset, cache):
        cfg = self.cfg
        B, L, _ = x.shape
        p = f"model.layers.{i}.self_attn"
        q = self._lin(f"{p}.q_proj", x).view(B, L, cfg.heads, cfg.head_dim).transpose(1, 2)
        k = self._lin(f"{p}.k_proj", x).view(B, L, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = self._lin(f"{p}.v_proj", x).view(B, L, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q = rms_norm(q, self.t[f"{p}.q_norm.weight"], cfg.eps)
        k = rms_norm(k, self.t[f"{p}.k_norm.weight"], cfg.eps)
        if i not in cfg.no_rope:
            q = rope_neox(q, offset, cfg.theta)
            k = rope_neox(k, offset, cfg.theta)
        if self.kv_store_dtype is not None and self.kv_store_dtype != k.dtype:
            k = k.to(self.kv_store_dtype).to(k.dtype)
            v = v.to(self.kv_store_dtype).to(v.dtype)
        sliding = cfg.layer_types[i] == "sliding_attention"
        entry = cache.layers[i] if cache is not None else None
        if sliding and cache is not None:
            if entry is None:
                entry = cache.layers[i] = SlidingKV(B, cfg.kv_heads, cfg.sliding_window, cfg.head_dim,
                                                    self.dtype, x.device)
            # keys = ring contents (positions < offset) + this chunk; mask by position.
            # A fresh ring (offset 0) holds nothing: attend to the chunk only.
            if offset == 0:
                kpos, keys, vals = qpos, k, v
            else:
                kpos = torch.cat([entry.pos, qpos])
                keys, vals = torch.cat([entry.k, k], dim=2), torch.cat([entry.v, v], dim=2)
            entry.write(k, v, qpos)
        else:
            if entry is not None:
                k = torch.cat([entry[0], k], dim=2)
                v = torch.cat([entry[1], v], dim=2)
            if cache is not None:
                cache.layers[i] = (k, v)
            keys, vals = k, v
            kpos = torch.arange(keys.shape[2], device=x.device)
        mask = (kpos[None, :] > qpos[:, None]) | (kpos[None, :] < 0)
        if sliding:
            mask = mask | (kpos[None, :] < qpos[:, None] - (cfg.sliding_window - 1))
        rep = cfg.heads // cfg.kv_heads
        kk = keys.repeat_interleave(rep, dim=1)
        vv = vals.repeat_interleave(rep, dim=1)
        scores = (q.float() @ kk.float().transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(mask, float("-inf"))
        sinks = self.t[f"{p}.softmax_offset"].float().view(1, -1, 1, 1).expand(B, cfg.heads, L, 1)
        probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1]
        out = (probs.to(self.dtype) @ vv).transpose(1, 2).reshape(B, L, cfg.heads * cfg.head_dim)
        return self._lin(f"{p}.o_proj", out)

    def _moe(self, x, i):
        cfg = self.cfg
        B, L, _ = x.shape
        pre = f"model.layers.{i}.mlp"
        xf = x.float()
        router_w = self.t[f"{pre}.gate.weight"].float()  # [E, H]
        bias = self.t[f"{pre}.gate.expert_bias"].float()
        gate = torch.sigmoid(F.linear(xf, router_w))  # [B, L, E]
        idx = torch.topk(gate + bias.view(1, 1, -1), cfg.topk, dim=-1).indices
        w_sel = torch.gather(gate, -1, idx)
        w_sel = w_sel / (w_sel.sum(-1, keepdim=True) + 1e-20)
        flat_x = x.reshape(-1, cfg.hidden)
        flat_idx = idx.reshape(-1, cfg.topk)
        flat_w = w_sel.reshape(-1, cfg.topk)
        out = torch.zeros_like(flat_x, dtype=torch.float32)
        wg = self.q[f"{pre}.switch_mlp.gate_proj"]  # [E, I, H]
        wu = self.q[f"{pre}.switch_mlp.up_proj"]
        wd = self.q[f"{pre}.switch_mlp.down_proj"]  # [E, H, I]
        # Dequantize only routed experts (decode touches top-k of E; one host
        # sync for the expert set per MoE layer).
        for e in torch.unique(flat_idx).tolist():
            hit = flat_idx == e  # [N, topk]
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            h = flat_x[rows]
            d = F.linear(F.silu(F.linear(h, wg.expert(e, self.dtype))) * F.linear(h, wu.expert(e, self.dtype)),
                         wd.expert(e, self.dtype))
            wght = (flat_w[rows] * hit[rows]).sum(-1, keepdim=True)
            out.index_add_(0, rows, d.float() * wght)
        out = out.to(self.dtype).view(B, L, cfg.hidden)
        return out + self._mlp(x, f"{pre}.shared_expert")

    # -- public ------------------------------------------------------------
    def forward(self, ids: torch.Tensor, kv: KVCache | None = None) -> torch.Tensor:
        """ids: [B, L] int64 on device. ``kv`` (from :meth:`new_cache`) is
        updated in place for incremental decode. Returns fp32 logits [B, L, V]."""
        cfg = self.cfg
        B, L = ids.shape
        offset = kv.offset if kv is not None else 0
        qpos = torch.arange(offset, offset + L, device=ids.device)
        h = self.embed.rows(ids, self.dtype)
        for i in range(cfg.layers):
            a = self._attention(rms_norm(h, self.t[f"model.layers.{i}.input_layernorm.weight"], cfg.eps),
                                i, qpos, offset, kv)
            h = h + a
            x = rms_norm(h, self.t[f"model.layers.{i}.post_attention_layernorm.weight"], cfg.eps)
            h = h + (self._mlp(x, f"model.layers.{i}.mlp") if i in cfg.dense_layers else self._moe(x, i))
        if kv is not None:
            kv.offset += L
        h = rms_norm(h, self.t["model.norm.weight"], cfg.eps)
        return F.linear(h, self.q["lm_head"].dense(self.dtype)).float()

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg)


def load_model(ckpt: Path, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> Model:
    """Load an MLX affine-quantized checkpoint dir (config.json + model-*.safetensors).

    Device layout is the contract in docs/rocm-model-layer.md: packed weights
    as int32 words, scales/biases in checkpoint dtype, dense params fp32.
    """
    ckpt = Path(ckpt)
    config = json.loads((ckpt / "config.json").read_text())
    cfg = Config.from_json(ckpt / "config.json")
    raw: dict[str, StTensor] = {}
    for p in sorted(ckpt.glob("model-*.safetensors")):
        shard = load_safetensors(p)
        if raw.keys() & shard.keys():
            raise ValueError(f"Duplicate tensor in {p}")
        raw.update(shard)
    assert_router_nondegenerate(
        {k: v.float32() for k, v in raw.items() if k.endswith(".mlp.gate.weight")}, config
    )
    dev = torch.device(device)
    qbits = config["mlx_export"]
    routed_bits = qbits["routed_expert_bits"]
    embed_bits = qbits["embedding_bits"]
    qlin: dict[str, QLinear] = {}
    dense: dict[str, torch.Tensor] = {}
    embed: QLinear | None = None
    for name, st in raw.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if name.endswith(".weight") and (name[:-len(".weight")] + ".scales") in raw:
            base = name[:-len(".weight")]
            bits = embed_bits if base == "model.embed_tokens" else (
                routed_bits if ".switch_mlp." in base else qbits["bits"])
            if st.dtype not in ("U8", "U32", "I32"):
                raise ValueError(f"{name}: packed weight must be U8/U32, got {st.dtype}")
            words = np.ascontiguousarray(st.data).view(np.int32)  # U8 [.., K/2] -> int32 [.., K/8]
            ql = QLinear(
                torch.from_numpy(words).to(dev),
                raw[base + ".scales"].torch(dev),
                raw[base + ".biases"].torch(dev),
                bits,
            )
            if base == "model.embed_tokens":
                embed = ql
            else:
                qlin[base] = ql
        else:
            dense[name] = st.torch(dev, torch.float32)
    if embed is None:
        raise ValueError("checkpoint has no quantized model.embed_tokens")
    # Production Flint exports no expert_bias (router_expert_bias: false); MLX
    # sanitize() fills zeros, so do the same.
    for i in range(cfg.layers):
        if i not in cfg.dense_layers:
            dense.setdefault(f"model.layers.{i}.mlp.gate.expert_bias",
                             torch.zeros(cfg.num_experts, dtype=torch.float32, device=dev))
    return Model(cfg, dense, qlin, embed, dev, dtype)
