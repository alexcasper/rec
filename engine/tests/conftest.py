"""Shared test fixtures and backend guards.

Test modules that exercise the resident Metal kernels (megakernel, KVStorage
banks, resident sampler) only run on the metal backend; they are skipped
automatically elsewhere so the Linux/CPU suite stays green.
"""

import json

import pytest

from rec_mlx.backends import detect_backend

METAL_ONLY_MODULES = {
    "test_serving.py",
    "test_grammar.py",
    "test_kernel_routing.py",
    "test_shared_kv.py",
    "test_generated_prefix.py",
    "test_prefix_rollback.py",
    "validate_resident.py",
    "validate_prefix_rollback.py",
}

if detect_backend().name != "metal":
    # Skip at collection time: these modules import resident-kernel machinery
    # (_grammar_native, Metal KV banks) that does not exist off the metal backend.
    collect_ignore_glob = list(METAL_ONLY_MODULES)


TINY_CONFIG = {
    "model_type": "flint",
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 513,
    "layer_types": ["sliding_attention", "full_attention"],
    "no_rope_layer_indices": [1],
    "mlp_only_layers": [0],
    "tie_word_embeddings": False,
    "kv_key_bits": 8,
    "kv_value_bits": 4,
    "kv_group_size": 64,
    "eos_token_id": [2],
    "quantization": {"group_size": 64, "bits": 4},
    "mlx_export": {"bits": 4, "routed_expert_bits": 4, "embedding_bits": 8},
}


def build_tiny_checkpoint(directory):
    """Materialize a valid tiny Flint checkpoint (random weights, real loader path)."""
    import mlx.core as mx
    import mlx.nn as nn

    from rec_mlx.model import Model, ModelArgs
    from rec_mlx.quantized import QATModel

    directory.mkdir(parents=True, exist_ok=True)
    config = dict(TINY_CONFIG)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(
        model,
        class_predicate=lambda path, module: {
            "group_size": 64,
            "bits": 8 if path == "model.embed_tokens" else 4,
            "mode": "affine",
        }
        if hasattr(module, "to_quantized")
        else False,
    )
    for i, layer in enumerate(model.layers):
        layer.self_attn.softmax_offset = mx.random.normal((config["num_attention_heads"],)) * 0.01
        if i not in config["mlp_only_layers"]:
            layer.mlp.gate.weight = mx.random.normal((config["num_experts"], config["hidden_size"])) * 0.05
            layer.mlp.gate.expert_bias = mx.random.normal((config["num_experts"],)) * 0.01
    flat = {}

    def flatten(prefix, params):
        for key, value in params.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flatten(path, value)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    flatten(f"{path}.{index}", item)
            elif isinstance(value, mx.array):
                flat[path] = value

    flatten("", dict(model.parameters()))
    mx.save_safetensors(str(directory / "model-00001.safetensors"), flat)

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {f"tok{i}": i for i in range(config["vocab_size"])}
    for index, token in enumerate(["<pad>", "<|begin_of_text|>", "<|end_of_text|>"]):
        vocabulary[token] = index
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|end_of_text|>",
        bos_token="<|begin_of_text|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    tokenizer.save_pretrained(directory)
    if not (directory / "special_tokens_map.json").is_file():
        (directory / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"})
        )
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model-00001.safetensors",
    ]
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "weight_format": "mlx-affine-int4",
                "kv_format": "bf16",
                "max_context": QATModel.context_limit,
                "model_id": "tiny-flint-test",
                "checkpoint_files": files,
            },
            indent=2,
        )
    )
    return directory


@pytest.fixture
def tiny_checkpoint(tmp_path):
    return build_tiny_checkpoint(tmp_path / "ckpt")
