"""rec-b3u: convert RecursiveSI/flint-int4 (MLX affine int4/int8, no quantization_config) to GGUF
with upstream llama.cpp's own FlintModel converter (a37cc46), dequantizing on the fly.

Only two adaptations are made, both upstream-converter-facing; the model math is untouched:
  1. every `<x>.weight` (U8 packed) that has `<x>.scales` + `<x>.biases` is replaced by a lazy
     dequant (bits inferred per tensor from shapes: embed_tokens is 8-bit, everything else 4-bit;
     group size 64). The dequant is engine/src/rec_mlx/torch_rocm.dequant's formulation, whose
     bit-identity is pinned by engine/tests/test_torch_rocm_dequant.py: low nibble first,
     fp32 q * s then + b, then cast to the output dtype;
  2. MLX-stacked experts `mlp.switch_mlp.{gate,up,down}_proj` [E, out, in] are renamed to the
     HF-merged form Qwen2MoeModel.modify_tensors already handles: gate+up concatenated on
     dim -2 into `mlp.experts.gate_up_proj` [E, 2*n_ff, n_embd], `mlp.experts.down_proj`
     [E, n_embd, n_ff] unchanged.

    PYTHONPATH=<llama.cpp>/gguf-py python convert_flint_int4.py <llama.cpp> <hf_dir> <out.gguf> [--outtype bf16]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

GROUP_SIZE = 64


def dequant(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """MLX affine dequant, U8 export bytes [..., K*bits/8] -> fp32 [..., K]."""
    b = packed.contiguous().view(torch.uint8)
    groups = scales.shape[-1]
    if biases.shape != scales.shape:
        raise ValueError(f"scales {tuple(scales.shape)} vs biases {tuple(biases.shape)}")
    bits = b.shape[-1] * 8 // (groups * GROUP_SIZE)
    if bits * groups * GROUP_SIZE != b.shape[-1] * 8 or bits not in (4, 8):
        raise ValueError(f"cannot infer bits: bytes {b.shape[-1]}, groups {groups}")
    q = torch.stack((b & 15, b >> 4), dim=-1) if bits == 4 else b
    q = q.reshape(*b.shape[:-1], groups, GROUP_SIZE).to(torch.float32)
    # Out-of-place on purpose: the converter's LazyTorchTensor silently drops in-place ops
    # (q.mul_(s).add_(b) wrote the raw 0..15 codes into the first GGUF). Same fp32 math: q*s, then +b.
    q = q * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return q.reshape(*b.shape[:-1], groups * GROUP_SIZE)


def main() -> int:
    llama, hf_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    sys.path.insert(0, str(llama))
    sys.path.insert(0, str(llama / "gguf-py"))
    from conversion.flint import FlintModel  # noqa: E402  (upstream, a37cc46)

    orig_dequant = FlintModel.dequant_model

    def dequant_model(self):
        mt = self.model_tensors
        n_deq = 0
        for name in [n for n in mt if n.endswith(".scales")]:
            base = name.removesuffix(".scales")
            w, s, bb = mt[base + ".weight"], mt[name], mt[base + ".biases"]
            mt[base + ".weight"] = lambda w=w, s=s, bb=bb: dequant(w(), s(), bb())
            del mt[name], mt[base + ".biases"]
            n_deq += 1
        n_exp = 0
        for name in [n for n in mt if ".mlp.switch_mlp.gate_proj.weight" in n]:
            pre = name.removesuffix("switch_mlp.gate_proj.weight")
            g, u, d = (mt.pop(f"{pre}switch_mlp.{p}_proj.weight") for p in ("gate", "up", "down"))
            mt[f"{pre}experts.gate_up_proj.weight"] = lambda g=g, u=u: torch.cat((g(), u()), dim=-2)
            mt[f"{pre}experts.down_proj.weight"] = d
            n_exp += 1
        left = [n for n in mt if "switch_mlp" in n or n.endswith((".scales", ".biases"))]
        if left:
            raise SystemExit(f"unconverted MLX tensors remain: {left[:5]}")
        print(f"[convert_flint_int4] dequantized {n_deq} MLX-affine tensors; merged experts in {n_exp} layers",
              flush=True)
        return orig_dequant(self)

    FlintModel.dequant_model = dequant_model
    import convert_hf_to_gguf  # noqa: E402

    args = [str(hf_dir), "--outfile", str(out)] + (sys.argv[4:] or ["--outtype", "bf16"])
    sys.argv = ["convert_hf_to_gguf.py"] + args
    return convert_hf_to_gguf.main() or 0


if __name__ == "__main__":
    sys.exit(main())
