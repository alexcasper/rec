"""Seeded synthetic Flint checkpoints (tests, CPU/GPU baselines, conformance).

Replaces the untracked ``engine/.build/cpu_baseline*.py`` exporters (rec-9as):
those started from ``Model()``'s zero-initialised ``Router`` and shipped an
all-zero MoE router, so every expert tied and cross-backend MoE comparisons
measured tie-breaking. Here every tensor the zero-init leaves degenerate
(attention sinks, router weight, expert bias) is drawn from a seeded RNG, and
``assert_router_nondegenerate`` runs before anything is written.

    python -m rec_mlx.synthetic --preset tiny --out engine/output/cpu-baseline-ckpt
    python -m rec_mlx.synthetic --preset mid  --out engine/output/cpu-baseline-mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_COMMON = {
    "model_type": "flint",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 513,
    "tie_word_embeddings": False,
    "kv_key_bits": 8,
    "kv_value_bits": 4,
    "kv_group_size": 64,
    "eos_token_id": [2],
    "quantization": {"group_size": 64, "bits": 4},
    "mlx_export": {"bits": 4, "routed_expert_bits": 4, "embedding_bits": 8},
}


def _mid_config(layers: int = 8) -> dict:
    return {
        **_COMMON,
        "hidden_size": 2048,
        "intermediate_size": 2048,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 1024,
        "num_hidden_layers": layers,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 32,
        "num_experts_per_tok": 2,
        "vocab_size": 150144,
        "layer_types": ["sliding_attention", "full_attention"] * (layers // 2),
        "no_rope_layer_indices": [i for i in range(layers) if i % 2 == 1],
        "mlp_only_layers": [0],
    }


PRESETS = {
    # 2 layers: L0 dense sliding+rope, L1 MoE (8 experts, top-2) full no-rope.
    # Same shapes as the HIP tiny megakernel.
    "tiny": {
        **_COMMON,
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
        "layer_types": ["sliding_attention", "full_attention"],
        "no_rope_layer_indices": [1],
        "mlp_only_layers": [0],
    },
    # Flint-shaped mid model: hidden 2048, vocab 150144, MoE 32/top-2, 8 layers.
    "mid": _mid_config(),
}

SPECIAL_TOKENS = ["<pad>", "<|begin_of_text|>", "<|end_of_text|>"]
CHAT_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>"]
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "model-00001.safetensors"]


def build_checkpoint(directory: Path, preset: str | dict = "tiny", seed: int = 0,
                     model_id: str | None = None) -> Path:
    """Write a valid synthetic checkpoint (config, tokenizer, weights, manifest).

    Weights: ``Model`` init under ``mx.random.seed(seed)``, quantized like
    production (INT4 affine g64, INT8 embeddings), then per layer the sinks
    (N(0, 0.01^2)) and, on MoE layers, router weight (N(0, 0.05^2)) and expert
    bias (N(0, 0.01^2)) are drawn in layer order.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from .checkpoint import assert_router_nondegenerate, validate
    from .model import Model, ModelArgs
    from .quantized import QATModel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(PRESETS[preset]) if isinstance(preset, str) else dict(preset)
    name = preset if isinstance(preset, str) else "custom"
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    mx.random.seed(seed)
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
    dense = set(config["mlp_only_layers"])
    for i, layer in enumerate(model.layers):
        layer.self_attn.softmax_offset = mx.random.normal((config["num_attention_heads"],)) * 0.01
        if i not in dense:
            layer.mlp.gate.weight = mx.random.normal((config["num_experts"], config["hidden_size"])) * 0.05
            layer.mlp.gate.expert_bias = mx.random.normal((config["num_experts"],)) * 0.01

    flat: dict = {}

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
    mx.eval(flat)
    assert_router_nondegenerate(
        {k: np.array(v.astype(mx.float32)) for k, v in flat.items() if k.endswith(".mlp.gate.weight")},
        config,
    )
    mx.save_safetensors(str(directory / "model-00001.safetensors"), flat)
    _write_tokenizer(directory, config["vocab_size"])
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "weight_format": "mlx-affine-int4",
                "kv_format": "bf16",
                "max_context": QATModel.context_limit,
                "model_id": model_id or f"synthetic-{name}-flint-seed{seed}",
                "checkpoint_files": FILES,
            },
            indent=2,
        )
    )
    validate(directory, directory / "manifest.json")
    return directory


def _write_tokenizer(directory: Path, vocab_size: int) -> None:
    """WordLevel tokenizer whose every id (incl. specials) is < vocab_size.

    Specials and chat tokens own ids 0..5 outright. An earlier version also
    mapped tok0..tok2 to 0..2: the duplicate ids made PreTrainedTokenizerFast
    re-register specials past the vocab (512, 513, ...) in a hash-order-
    dependent way, and the server's chat template then fed out-of-range ids
    to the model (a GPU hardware fault on rocm).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    reserved = SPECIAL_TOKENS + CHAT_TOKENS
    vocabulary = {token: index for index, token in enumerate(reserved)}
    vocabulary.update({f"tok{i}": i for i in range(len(reserved), vocab_size)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|end_of_text|>",
        bos_token="<|begin_of_text|>",
        additional_special_tokens=CHAT_TOKENS,
    )
    tokenizer.save_pretrained(directory)
    if not (directory / "special_tokens_map.json").is_file():
        (directory / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>",
                        "additional_special_tokens": CHAT_TOKENS})
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id")
    args = parser.parse_args(argv)
    path = build_checkpoint(args.out, args.preset, args.seed, args.model_id)
    print(f"wrote {path} ({args.preset}, seed {args.seed}); validate() OK, router non-degenerate")


if __name__ == "__main__":
    main()
