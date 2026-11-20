"""rec-xh7 incumbent row: write the mid-8L checkpoint (rec mid-prod, MLX affine INT4/INT8)
as an F16 GGUF that llama.cpp's flint arch loads.

Metadata + tokenizer: copied from the real flint-preview GGUF, with the shape keys taken
from the mid config (block_count, expert counts, dense FFN, leading dense blocks, attention
and rope patterns). Weights: dequantized with rec_mlx.torch_rocm.dequant (bit-identical to
MLX, pinned by tests/test_torch_rocm_dequant.py) and written F16 (norms/sinks/router F32)
under the real file's tensor names; orientation is the HF [out, in] = GGUF ne reversed.

    python build_mid_gguf.py REAL.gguf MID_CKPT_DIR OUT.gguf ENGINE_SRC
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

REAL, CKPT, OUT, SRC = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, SRC)
import torch  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
from rec_mlx.torch_rocm import dequant  # noqa: E402

cfg = json.loads((CKPT / "config.json").read_text())
L = cfg["num_hidden_layers"]
dense = set(cfg["mlp_only_layers"])
assert sorted(dense) == list(range(len(dense))), "flint arch needs leading dense blocks"

# ---- raw safetensors (single shard)
shard = CKPT / "model-00001.safetensors"
raw = shard.read_bytes()
n = struct.unpack("<Q", raw[:8])[0]
hdr = json.loads(raw[8:8 + n])
hdr.pop("__metadata__", None)
DT = {"BF16": np.uint16, "U8": np.uint8, "F32": np.float32, "U32": np.uint32, "F16": np.float16}


def arr(name):
    h = hdr[name]
    a, b = h["data_offsets"]
    return np.frombuffer(raw, dtype=DT[h["dtype"]], count=(b - a) // np.dtype(DT[h["dtype"]]).itemsize,
                         offset=8 + n + a).reshape(h["shape"]), h["dtype"]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def plain(name):
    a, dt = arr(name)
    return bf16_to_f32(a) if dt == "BF16" else a.astype(np.float32)


def weight(prefix):
    """dequantized float32 [..., out, in] for a packed MLX linear, else the plain tensor."""
    if prefix + ".scales" not in hdr:
        return plain(prefix + ".weight")
    packed, _ = arr(prefix + ".weight")
    s, b = plain(prefix + ".scales"), plain(prefix + ".biases")
    K = s.shape[-1] * 64
    bits = packed.shape[-1] * 8 // K
    words = torch.from_numpy(np.ascontiguousarray(packed).view(np.int32))
    w = dequant(words, torch.from_numpy(s).to(torch.bfloat16), torch.from_numpy(b).to(torch.bfloat16),
                bits, torch.float32)
    return w.numpy()


# ---- metadata
r = GGUFReader(REAL)
w = GGUFWriter(OUT, "flint")
pattern_sliding = [t == "sliding_attention" for t in cfg["layer_types"]]
pattern_rope = [i not in cfg["no_rope_layer_indices"] for i in range(L)]
override = {
    "flint.block_count": L,
    "flint.feed_forward_length": cfg["intermediate_size"],
    "flint.expert_count": cfg["num_experts"],
    "flint.expert_used_count": cfg["num_experts_per_tok"],
    "flint.leading_dense_block_count": len(dense),
    "flint.expert_feed_forward_length": cfg["moe_intermediate_size"],
    "flint.expert_shared_feed_forward_length": cfg["shared_expert_intermediate_size"],
    "flint.expert_shared_count": cfg["shared_expert_intermediate_size"] // cfg["moe_intermediate_size"],
    "flint.attention.sliding_window": cfg["sliding_window"],
    "flint.attention.sliding_window_pattern": pattern_sliding,
    "flint.attention.rope_pattern": pattern_rope,
    "flint.embedding_length": cfg["hidden_size"],
    "flint.attention.head_count": cfg["num_attention_heads"],
    "flint.attention.head_count_kv": cfg["num_key_value_heads"],
    "flint.attention.key_length": cfg["head_dim"],
    "flint.attention.value_length": cfg["head_dim"],
    "general.name": "Flint mid-8L synthetic (rec mid-prod, seed 0)",
}
skip_prefix = ("GGUF.", "quantize.", "general.file_type", "general.architecture")
for key, f in r.fields.items():
    if key.startswith(skip_prefix):
        continue
    vt = f.types[0]
    sub = f.types[1] if vt == GGUFValueType.ARRAY and len(f.types) > 1 else None
    if key in override:
        val = override[key]
    elif vt == GGUFValueType.ARRAY:
        if sub == GGUFValueType.STRING:
            val = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        else:
            val = [f.parts[i].tolist()[0] for i in f.data]
    elif vt == GGUFValueType.STRING:
        val = bytes(f.parts[f.data[0]]).decode("utf-8")
    else:
        val = f.parts[f.data[0]].tolist()[0]
    if key == "general.name":
        w.add_key_value(key, val, vt)
    else:
        w.add_key_value(key, val, vt, sub_type=sub)
missing = set(override) - set(r.fields)
assert not missing, f"override keys absent from the real GGUF: {missing}"
w.add_file_type(1)  # MOSTLY_F16

# ---- tensors
assert cfg["vocab_size"] == len(r.fields["tokenizer.ggml.tokens"].data), "tokenizer/vocab mismatch"


def put(name, a, f32=False):
    a = np.ascontiguousarray(a.astype(np.float32 if f32 else np.float16))
    w.add_tensor(name, a, raw_dtype=GGMLQuantizationType.F32 if f32 else GGMLQuantizationType.F16)


put("token_embd.weight", weight("model.embed_tokens"))
put("output_norm.weight", plain("model.norm.weight"), f32=True)
put("output.weight", weight("lm_head"))
for i in range(L):
    p, b = f"model.layers.{i}", f"blk.{i}"
    put(f"{b}.attn_norm.weight", plain(f"{p}.input_layernorm.weight"), f32=True)
    put(f"{b}.ffn_norm.weight", plain(f"{p}.post_attention_layernorm.weight"), f32=True)
    for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
        put(f"{b}.{gg}.weight", weight(f"{p}.self_attn.{hf}"))
    put(f"{b}.attn_q_norm.weight", plain(f"{p}.self_attn.q_norm.weight"), f32=True)
    put(f"{b}.attn_k_norm.weight", plain(f"{p}.self_attn.k_norm.weight"), f32=True)
    put(f"{b}.attn_sinks.weight", plain(f"{p}.self_attn.softmax_offset"), f32=True)
    if i in dense:
        for hf in ("gate", "up", "down"):
            put(f"{b}.ffn_{hf}.weight", weight(f"{p}.mlp.{hf}_proj"))
    else:
        put(f"{b}.ffn_gate_inp.weight", plain(f"{p}.mlp.gate.weight"), f32=True)
        assert f"{p}.mlp.gate.expert_bias" not in hdr
        for hf in ("gate", "up", "down"):
            put(f"{b}.ffn_{hf}_exps.weight", weight(f"{p}.mlp.switch_mlp.{hf}_proj"))
            put(f"{b}.ffn_{hf}_shexp.weight", weight(f"{p}.mlp.shared_expert.{hf}_proj"))
used = {k for k in hdr}
print("tensors written; source tensors:", len(used))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", OUT)
