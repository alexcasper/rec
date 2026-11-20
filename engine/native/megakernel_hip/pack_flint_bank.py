"""Config-driven bank packer for the full Flint checkpoint (rec-xh7 prep).

Reads an mlx-affine INT4/INT8 checkpoint (HF RecursiveSI/flint-int4 layout:
U8 packed [.., out, K/2], BF16 scales/biases [.., out, K/64], INT8 embed,
no expert_bias) and packs it into the zero-copy byte bank + offset table
consumed by the HIP megakernel drivers, mirroring pack_mid_bank's ordering
parameterized by config.json. Streams shard->bank tensor by tensor: peak
RAM is one tensor, never the 21+ GiB bank.

Layout contract: docs/rocm-model-layer.md s1-s2.
  - packed weights: raw bytes, zero-copy (U8 and U32 little-endian are
    byte-identical for this layout)
  - scales/biases/norms/router: BF16 uint16 bits (rounded from F32 if needed)
  - expert strides downstream: BYTES for packed, ELEMENTS for bf16 s/b
  - embed: INT8 affine qlinear (bits=8)

Ordering (offset table), parameterized by config:
  [0 .. 7*NL-1]        per layer: input_ln, post_ln, q_norm, k_norm,
                       softmax_offset, router weight (dense layers: none),
                       expert_bias (absent in real Flint -> zeros(E))
  [7*NL]               final norm
  [7*NL+1]             embed qlinear (INT8)
  [7*NL+2 ..]          per layer q4 block:
                         dense layer: q,k,v,o + gate,up,down + 3 empty
                         MoE layer:   q,k,v,o + switch gate,up,down +
                                      shared gate,up,down
  [last]               total bank byte length sentinel

Verify mode (--verify) re-reads the packed bank and checks byte identity
of every tensor region against a fresh range-read of the source shards,
and dequant identity (q*s+b, BF16-rounded s/b) on a per-tensor basis in
row chunks. Cotenant state (dusk llama servers + gpu_busy) is logged
with every progress line per the benchmark hygiene protocol.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ST_DTYPES = {
    "BF16": np.uint16, "F32": np.float32, "F64": np.float64,
    "U32": np.uint32, "I32": np.int32, "U8": np.uint8,
    "I8": np.int8, "U16": np.uint16,
}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def cotenant_state() -> str:
    """Dusk llama cotenants + GPU busy, for hygiene logging."""
    try:
        p = subprocess.run(
            ["bash", "-c",
             "for pat in llama-server flint qwen; do pgrep -fc $pat 2>/dev/null; done; "
             "cat /sys/class/drm/card[0-9]/device/gpu_busy_percent 2>/dev/null"],
            capture_output=True, text=True, timeout=10)
        counts = p.stdout.split()
        if len(counts) >= 4:
            return (f"cotenants[flint={counts[0]},e4b={counts[1]},qwen={counts[2]}] "
                    f"gpu_busy={counts[3]}%")
    except Exception:
        pass
    return "cotenants[unknown]"


class ShardIndex:
    """Random-access index over all model-*.safetensors shards."""

    def __init__(self, ckpt_dir: Path):
        self.tensors: dict[str, dict] = {}
        for p in sorted(Path(ckpt_dir).glob("model-*.safetensors")):
            with open(p, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
                base = 8 + n
                header.pop("__metadata__", None)
                for name, info in header.items():
                    if name in self.tensors:
                        raise ValueError(f"duplicate tensor {name}")
                    start, end = info["data_offsets"]
                    self.tensors[name] = {
                        "path": p, "dtype": info["dtype"], "shape": info["shape"],
                        "abs_start": base + start, "nbytes": end - start,
                    }
        log(f"shard index: {len(self.tensors)} tensors, "
            f"{sum(t['nbytes'] for t in self.tensors.values()) / 2**30:.2f} GiB")

    def read(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        with open(t["path"], "rb") as f:
            f.seek(t["abs_start"])
            buf = f.read(t["nbytes"])
        return np.frombuffer(buf, dtype=ST_DTYPES[t["dtype"]]).reshape(t["shape"])

    def read_bytes(self, name: str) -> bytes:
        t = self.tensors[name]
        with open(t["path"], "rb") as f:
            f.seek(t["abs_start"])
            return f.read(t["nbytes"])

    def read_slice(self, name: str, r0: int, r1: int) -> np.ndarray:
        """Read a contiguous row-range [r0, r1) of a tensor, FLAT (row-major):
        returns (r1-r0, trailing-storage-units). Row-wise dequant only needs
        the last dim's units, and r0/r1 need not align with shape[0]."""
        t = self.tensors[name]
        row = t["nbytes"] // max(1, int(np.prod(t["shape"][:-1])))
        with open(t["path"], "rb") as f:
            f.seek(t["abs_start"] + r0 * row)
            buf = f.read((r1 - r0) * row)
        return np.frombuffer(buf, dtype=ST_DTYPES[t["dtype"]]).reshape(r1 - r0, t["shape"][-1])


def f32_to_bf16(v: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)


def to_bf16_bits(arr: np.ndarray, dtype_tag: str) -> np.ndarray:
    if dtype_tag == "BF16":
        return np.ascontiguousarray(arr, dtype=np.uint16)
    return f32_to_bf16(arr.astype(np.float32))


def plan_offsets(cfg: dict) -> list[dict]:
    """Build the ordered list of bank entries from config.json."""
    nl = cfg["num_hidden_layers"]
    ne = cfg["num_experts"]
    dense = set(cfg.get("mlp_only_layers", []))
    entries: list[dict] = []
    for l in range(nl):
        pre = f"model.layers.{l}"
        entries += [
            {"kind": "bf16", "name": f"{pre}.input_layernorm.weight"},
            {"kind": "bf16", "name": f"{pre}.post_attention_layernorm.weight"},
            {"kind": "bf16", "name": f"{pre}.self_attn.q_norm.weight"},
            {"kind": "bf16", "name": f"{pre}.self_attn.k_norm.weight"},
            {"kind": "bf16", "name": f"{pre}.self_attn.softmax_offset"},
        ]
        if l in dense:
            entries += [{"kind": "bf16_none"}, {"kind": "bf16_none"}]
        else:
            entries += [
                {"kind": "bf16", "name": f"{pre}.mlp.gate.weight"},
                {"kind": "bf16", "name": f"{pre}.mlp.gate.expert_bias",
                 "fallback": ne},
            ]
    entries.append({"kind": "bf16", "name": "model.norm.weight"})
    entries.append({"kind": "qlinear", "name": "model.embed_tokens", "bits": 8})
    for l in range(nl):
        ap = f"model.layers.{l}.self_attn"
        mp = f"model.layers.{l}.mlp"
        entries += [
            {"kind": "qlinear", "name": f"{ap}.q_proj", "bits": 4},
            {"kind": "qlinear", "name": f"{ap}.k_proj", "bits": 4},
            {"kind": "qlinear", "name": f"{ap}.v_proj", "bits": 4},
            {"kind": "qlinear", "name": f"{ap}.o_proj", "bits": 4},
        ]
        if l in dense:
            entries += [
                {"kind": "qlinear", "name": f"{mp}.gate_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.up_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.down_proj", "bits": 4},
            ] + [{"kind": "empty"}] * 3
        else:
            entries += [
                {"kind": "qlinear", "name": f"{mp}.switch_mlp.gate_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.switch_mlp.up_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.switch_mlp.down_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.shared_expert.gate_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.shared_expert.up_proj", "bits": 4},
                {"kind": "qlinear", "name": f"{mp}.shared_expert.down_proj", "bits": 4},
            ]
    entries.append({"kind": "qlinear", "name": "lm_head", "bits": 4})
    entries.append({"kind": "sentinel"})
    return entries


def pack(ckpt_dir: Path, out_bank: Path, out_off: Path) -> None:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    idx = ShardIndex(ckpt_dir)
    entries = plan_offsets(cfg)
    log(f"plan: {len(entries)} entries, NL={cfg['num_hidden_layers']}, "
        f"E={cfg['num_experts']}, topk={cfg['num_experts_per_tok']} "
        f"| {cotenant_state()}")
    missing = []
    for e in entries:
        if e["kind"] not in ("bf16", "qlinear"):
            continue
        name = e["name"] + ".weight" if e["kind"] == "qlinear" else e["name"]
        has_fallback = e["kind"] == "bf16" and "fallback" in e
        if not has_fallback and name not in idx.tensors and e["name"] not in idx.tensors:
            missing.append(e["name"])
    if missing:
        raise KeyError(f"tensors missing from shards: {missing[:8]}...")

    offsets: list[int] = []
    written = 0
    with open(out_bank, "wb") as out:
        def align2():
            nonlocal written
            if written & 1:
                out.write(b"\x00")
                written += 1

        for i, e in enumerate(entries):
            kind = e["kind"]
            if kind == "sentinel":
                align2()
                offsets.append(written)
                continue
            align2()
            offsets.append(written)
            if kind in ("empty", "bf16_none"):
                continue
            if kind == "bf16" and e["name"] not in idx.tensors:
                n = e.get("fallback", 0)
                out.write(np.zeros(n, dtype=np.uint16).tobytes())
                written += n * 2
                continue
            if kind == "bf16":
                arr = idx.read(e["name"])
                bits = to_bf16_bits(arr, idx.tensors[e["name"]]["dtype"])
                out.write(np.ascontiguousarray(bits).tobytes())
                written += bits.nbytes
            else:  # qlinear
                s_name, b_name = e["name"] + ".scales", e["name"] + ".biases"
                w_bytes = idx.read_bytes(e["name"] + ".weight")
                s_bits = to_bf16_bits(idx.read(s_name), idx.tensors[s_name]["dtype"])
                b_bits = to_bf16_bits(idx.read(b_name), idx.tensors[b_name]["dtype"])
                out.write(w_bytes)
                out.write(np.ascontiguousarray(s_bits).tobytes())
                out.write(np.ascontiguousarray(b_bits).tobytes())
                written += len(w_bytes) + s_bits.nbytes + b_bits.nbytes
            if i % 64 == 0:
                log(f"packed {i}/{len(entries)} written={written / 2**30:.2f} GiB "
                    f"| {cotenant_state()}")
    off_arr = np.array(offsets, dtype=np.int64)
    off_arr.tofile(out_off)
    log(f"DONE bank={out_bank} ({written / 2**30:.3f} GiB) offsets={out_off} "
        f"({len(offsets)} entries) | {cotenant_state()}")


def packed_rows_k(shape: tuple, dtype_tag: str, bits: int) -> tuple[int, int]:
    """(rows, K) for a packed qlinear weight of any rank: rows = prod(shape[:-1])."""
    per_unit = 8 // bits if dtype_tag in ("U8", "I8") else 32 // bits
    rows = int(np.prod(shape[:-1]))
    return rows, shape[-1] * per_unit


def dequant_row_chunk(bank: np.ndarray, off: int, rows: int, k: int, bits: int,
                      r0: int, r1: int) -> np.ndarray:
    """Dequantize rows [r0, r1) of one qlinear bank region, straight from the
    (possibly memmapped) bank. Region layout: packed [rows, k*bits/8] |
    bf16 s [rows, k/64] | bf16 b [rows, k/64]. BF16-rounded s/b, low nibble =
    even element (INT4) — mirrors mid_mk_driver.dequant_np."""
    w_row = k * bits // 8
    sb_row = (k // 64) * 2
    s_off = off + rows * w_row
    b_off = s_off + rows * sb_row
    w = np.asarray(bank[off + r0 * w_row:off + r1 * w_row]).reshape(r1 - r0, k * bits // 8)
    s = np.asarray(bank[s_off + r0 * sb_row:s_off + r1 * sb_row]).view(np.uint16).reshape(r1 - r0, k // 64)
    b = np.asarray(bank[b_off + r0 * sb_row:b_off + r1 * sb_row]).view(np.uint16).reshape(r1 - r0, k // 64)
    s = (s.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    b = (b.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    if bits == 8:
        q = w.astype(np.float64)
    else:
        q = np.empty((r1 - r0, k), dtype=np.float64)
        q[:, 0::2] = (w & 0x0F).astype(np.float64)
        q[:, 1::2] = ((w >> 4) & 0x0F).astype(np.float64)
    s_rep = np.repeat(s, 64, axis=-1)
    b_rep = np.repeat(b, 64, axis=-1)
    return q * s_rep + b_rep


def dequant_source_row_chunk(idx: ShardIndex, name: str, bits: int,
                             r0: int, r1: int) -> np.ndarray:
    """Dequantize rows [r0, r1) of a source qlinear tensor (ranged reads)."""
    wname, sname, bname = (name + ".weight", name + ".scales", name + ".biases")
    w = idx.read_slice(wname, r0, r1)
    s_bits = to_bf16_bits(idx.read_slice(sname, r0, r1), idx.tensors[sname]["dtype"])
    b_bits = to_bf16_bits(idx.read_slice(bname, r0, r1), idx.tensors[bname]["dtype"])
    s = (s_bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    b = (b_bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    w_u8 = np.ascontiguousarray(w).view(np.uint8)
    if bits == 8:
        q = w_u8.astype(np.float64)
    else:
        low = (w_u8 & 0x0F).astype(np.float64)
        high = ((w_u8 >> 4) & 0x0F).astype(np.float64)
        q = np.stack([low, high], axis=-1).reshape(r1 - r0, -1)
    s_rep = np.repeat(s, 64, axis=-1)
    b_rep = np.repeat(b, 64, axis=-1)
    return q * s_rep + b_rep


def verify(ckpt_dir: Path, bank_path: Path, off_path: Path) -> int:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    idx = ShardIndex(ckpt_dir)
    entries = plan_offsets(cfg)
    offsets = np.fromfile(off_path, dtype=np.int64)
    assert len(offsets) == len(entries), "offset table length mismatch"
    # memmap, not fromfile: the bank can exceed MemAvailable on a cotenant-loaded
    # host (21 GiB bank vs ~11 GiB available); pages fault in per region.
    bank = np.memmap(bank_path, dtype=np.uint8, mode="r")
    log(f"verify: bank {bank.nbytes / 2**30:.3f} GiB (memmap), {len(entries)} entries "
        f"| {cotenant_state()}")
    failures = 0
    for i, e in enumerate(entries):
        if e["kind"] not in ("bf16", "qlinear"):
            continue
        off = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else int(offsets[-1])
        # byte identity: bank region must equal source bytes (packed w + bf16 s/b)
        if e["kind"] == "bf16":
            if e["name"] not in idx.tensors:
                src = np.zeros(e.get("fallback", 0), dtype=np.uint16).tobytes()
            else:
                src = to_bf16_bits(idx.read(e["name"]), idx.tensors[e["name"]]["dtype"]).tobytes()
        else:
            s_name, b_name = e["name"] + ".scales", e["name"] + ".biases"
            src = (idx.read_bytes(e["name"] + ".weight")
                   + to_bf16_bits(idx.read(s_name), idx.tensors[s_name]["dtype"]).tobytes()
                   + to_bf16_bits(idx.read(b_name), idx.tensors[b_name]["dtype"]).tobytes())
        region = bank[off:off + len(src)]
        if region.tobytes() != src:
            print(f"BYTE MISMATCH {e['name']} at offset {off}", flush=True)
            failures += 1
            continue
        if e["kind"] == "qlinear":
            wname = e["name"] + ".weight"
            wt = idx.tensors[wname]
            rows, k = packed_rows_k(wt["shape"], wt["dtype"], e["bits"])
            mism = False
            chunk = max(1, min(rows, 2**21 // max(k, 1)))
            for r0 in range(0, rows, chunk):
                r1 = min(r0 + chunk, rows)
                got = dequant_row_chunk(bank, off, rows, k, e["bits"], r0, r1)
                want = dequant_source_row_chunk(idx, e["name"], e["bits"], r0, r1)
                if not np.array_equal(got, want):
                    diff = float(np.abs(got - want).max())
                    print(f"DEQUANT MISMATCH {e['name']} rows[{r0}:{r1}] maxabs={diff}",
                          flush=True)
                    mism = True
                    failures += 1
                    break
            if mism:
                continue
        if i % 32 == 0:
            log(f"verified {i}/{len(entries)} | {cotenant_state()}")
    log(f"VERIFY {'PASS' if failures == 0 else f'FAIL ({failures})'} "
        f"| {cotenant_state()}")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-bank", type=Path)
    ap.add_argument("--out-off", type=Path)
    ap.add_argument("--verify-bank", type=Path)
    ap.add_argument("--verify-off", type=Path)
    a = ap.parse_args()
    if a.verify_bank:
        return verify(a.ckpt, a.verify_bank, a.verify_off)
    if not (a.out_bank and a.out_off):
        ap.error("--out-bank/--out-off or --verify-bank/--verify-off required")
    pack(a.ckpt, a.out_bank, a.out_off)
    return 0


if __name__ == "__main__":
    sys.exit(main())
