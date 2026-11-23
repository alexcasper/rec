"""Convert ops_rec synthetic mid-8L checkpoint (MLX affine INT4/INT8) to flint-arch GGUF F16.

Support-seat work for flint-y02: same-shape llama.cpp arm vs ops_rec mid-8L runs.
Tensor naming per llama.cpp src/models/flint.cpp (a37cc46, PR #30928).
Run with the rec e2e venv (has mlx): ~/rec-ve/.venv-e2e/bin/python convert_mid_gguf.py
"""
import json
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, "/home/dusk/llama.cpp/gguf-py")
import gguf  # noqa: E402

CKPT = "/home/dusk/rec-ve/ckpt-e2e/mid"
OUT = "/home/dusk/flint-mid/flint-mid-8L-f16.gguf"

cfg = json.loads(open(f"{CKPT}/config.json").read())
raw = {}
for shard in sorted(__import__("glob").glob(f"{CKPT}/model-*.safetensors")):
    raw.update({k: v for k, v in mx.load(shard).items() if v is not None})

GS = 64


def dequant(name):
    """Dequantize `<base>.weight` with siblings .scales/.biases; else pass through."""
    if not name.endswith(".weight") or f"{name[:-7]}.scales" not in raw:
        return np.array(raw[name])
    base = name[:-7]
    w = raw[name]
    bits = 8 if "embed_tokens" in name else 4
    d = mx.dequantize(w, raw[f"{base}.scales"], raw[f"{base}.biases"],
                      group_size=GS, bits=bits)
    return np.array(d.astype(mx.float32))


def out_name(name):
    n = name
    n = n.replace("model.embed_tokens", "token_embd")
    n = n.replace("lm_head", "output")
    n = n.replace("model.norm", "output_norm")
    n = n.replace("input_layernorm", "attn_norm")
    n = n.replace("post_attention_layernorm", "ffn_norm")
    n = n.replace("self_attn.q_proj", "attn_q").replace("self_attn.k_proj", "attn_k")
    n = n.replace("self_attn.v_proj", "attn_v").replace("self_attn.o_proj", "attn_output")
    n = n.replace("self_attn.q_norm", "attn_q_norm").replace("self_attn.k_norm", "attn_k_norm")
    n = n.replace("self_attn.softmax_offset", "attn_sinks.weight")
    n = n.replace("mlp.switch_mlp.gate_proj", "ffn_gate_exps")
    n = n.replace("mlp.switch_mlp.up_proj", "ffn_up_exps")
    n = n.replace("mlp.switch_mlp.down_proj", "ffn_down_exps")
    n = n.replace("mlp.shared_expert.gate_proj", "ffn_gate_shexp")
    n = n.replace("mlp.shared_expert.up_proj", "ffn_up_shexp")
    n = n.replace("mlp.shared_expert.down_proj", "ffn_down_shexp")
    n = n.replace("mlp.gate.weight", "ffn_gate_inp.weight")
    n = n.replace("mlp.gate.expert_bias", "exp_probs_b.bias")
    n = n.replace("mlp.gate_proj", "ffn_gate").replace("mlp.up_proj", "ffn_up")
    n = n.replace("mlp.down_proj", "ffn_down")
    return n.replace("model.layers.", "blk.")


L = cfg["num_hidden_layers"]
dense = set(cfg["mlp_only_layers"])

w = gguf.GGUFWriter(OUT, "flint")
w.add_name("flint-mid-8L-synthetic")
w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

w.add_uint32("flint.block_count", L)
w.add_uint32("flint.context_length", 16384)
w.add_uint32("flint.embedding_length", cfg["hidden_size"])
w.add_uint32("flint.feed_forward_length", cfg["intermediate_size"])
w.add_uint32("flint.attention.head_count", cfg["num_attention_heads"])
w.add_uint32("flint.attention.head_count_kv", cfg["num_key_value_heads"])
w.add_uint32("flint.attention.key_length", cfg["head_dim"])
w.add_uint32("flint.attention.value_length", cfg["head_dim"])
w.add_float32("flint.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
w.add_float32("flint.rope.freq_base", cfg["rope_theta"])
w.add_uint32("flint.attention.sliding_window", cfg["sliding_window"])
swa = [1 if t == "sliding_attention" else 0 for t in cfg["layer_types"]]
rope = [0 if i in cfg["no_rope_layer_indices"] else 1 for i in range(L)]
from gguf import GGUFValueType
U32ARR = (GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
w.add_key_value("flint.attention.sliding_window_pattern", swa, *U32ARR)
w.add_key_value("flint.attention.rope_pattern", rope, *U32ARR)
w.add_uint32("flint.leading_dense_block_count", len(dense))
w.add_uint32("flint.expert_count", cfg["num_experts"])
w.add_uint32("flint.expert_used_count", cfg["num_experts_per_tok"])
w.add_uint32("flint.expert_shared_count", 1)
w.add_float32("flint.expert_weights_scale", 1.0)
w.add_bool("flint.expert_weights_norm", True)
w.add_uint32("flint.expert_gating_func", 2)  # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
w.add_key_value("flint.expert_feed_forward_length",
                # all entries = expert FFN width: upstream flint.cpp reads entry[0]
                # (hparams.n_ff_exp() default il=0) once for every MoE layer
                [cfg["moe_intermediate_size"]] * L, *U32ARR)
w.add_uint32("flint.expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"])

# Tokenizer (synthetic WordLevel vocab, token->id in tokenizer.json)
tk = json.loads(open(f"{CKPT}/tokenizer.json").read())
vocab = tk["model"]["vocab"]
specials = set(a["content"] for a in tk.get("added_tokens", []))
ids = sorted(vocab, key=lambda t: vocab[t])
assert len(ids) == cfg["vocab_size"] and all(vocab[t] == i for i, t in enumerate(ids))
w.add_tokenizer_model("gpt2")  # WordLevel vocab; SPM ("llama") wants byte tokens we don't have
w.add_token_list(ids)
w.add_token_scores([0.0] * len(ids))
w.add_token_types([gguf.constants.TokenType.CONTROL if t in specials else gguf.constants.TokenType.NORMAL
                   for t in ids])
# BPE branch requires a merges KV; supply one inert pair (eos+bor adjacent never occurs;
# prompts are all-bos special-token runs with no byte/char tokens needed)
w.add_token_merges(["<|end_of_text|> <|begin_of_text|>"])
w.add_bos_token_id(0)
w.add_eos_token_id(2)
w.add_add_bos_token(False)

# Tensors: 2D -> F16, 1-D norms/sinks/router -> F32, 3D expert stacks -> F16
seen = set()
for name in sorted(raw):
    if not (name.endswith(".weight") or name.endswith(".expert_bias")
            or name.endswith("softmax_offset")):
        continue
    arr = dequant(name if name.endswith(".weight") else name)
    on = out_name(name)
    if on in seen:
        raise SystemExit(f"duplicate {on}")
    seen.add(on)
    if arr.ndim >= 2 and "ffn_exp_probs_b" not in on and "ffn_gate_inp" not in on:
        arr = arr.astype(np.float16)
    print(f"{on:48s} {str(arr.shape):20s} {arr.dtype}")
    w.add_tensor(on, arr)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("WROTE", OUT)
