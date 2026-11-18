import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def assert_router_nondegenerate(weights: Mapping[str, np.ndarray], config: dict) -> None:
    """Reject MoE layers whose router cannot distinguish experts (rec-9as).

    ``Router`` zero-initialises ``gate.weight``; an export that never writes a
    trained (or seeded) router ships all-zero rows. Then every expert scores
    sigmoid(0) = 0.5, top-k is decided by backend tie-breaking (MLX
    argpartition vs torch topk vs a HIP kernel pick different experts) and
    cross-backend outputs diverge at the first MoE layer. Identical rows are
    the general case of the same failure. ``expert_bias`` alone is not
    checked: zeros there are legitimate (sanitize() fills it in).

    ``weights`` maps tensor names to float numpy arrays; only
    ``*.mlp.gate.weight`` entries are read. Missing tensors are left to the
    strict loader to report.
    """
    dense = set(config.get("mlp_only_layers", []))
    for index in range(config["num_hidden_layers"]):
        if index in dense:
            continue
        name = f"model.layers.{index}.mlp.gate.weight"
        if name not in weights:
            continue
        w = np.asarray(weights[name], dtype=np.float32)
        if w.ndim != 2 or w.shape[0] < 2:
            continue
        if not np.any(w != w[:1]):
            kind = "all-zero" if not np.any(w) else "identical rows"
            raise ValueError(
                f"{name}: degenerate MoE router ({kind}); every expert ties and top-k "
                "routing becomes backend tie-breaking. Export a trained or seeded router (rec-9as)."
            )


def validate(directory: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["weight_format"] != "mlx-affine-int4" or manifest["kv_format"] != "bf16":
        raise ValueError("This engine requires affine INT4 weights and BF16 KV")
    files = manifest["checkpoint_files"]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    shards = {name for name in files if name.startswith("model-") and name.endswith(".safetensors")}
    if not required.issubset(files) or not shards:
        raise ValueError("Manifest must include configuration, tokenizer and weight shards")
    # The upstream loader discovers shards by glob; an unpinned extra shard must not be loaded.
    if {p.name for p in directory.glob("model-*.safetensors")} != shards:
        raise ValueError("Checkpoint shards do not match the manifest")
    # rec verifies every file's SHA-256 before it starts the engine.
    for name in files:
        if Path(name).name != name:
            raise ValueError(f"Checkpoint filename must be a basename: {name}")
        if not (directory / name).is_file():
            raise ValueError(f"Checkpoint is missing a manifest file: {name}")
    config = json.loads((directory / "config.json").read_text())
    if config["sliding_window"] != 513:
        raise ValueError("Flint requires sliding_window=513, including the current token")
    export = config["mlx_export"]
    if (export["bits"], export["routed_expert_bits"], export["embedding_bits"]) != (4, 4, 8):
        raise ValueError("Expected INT4 projections/experts and INT8 embeddings")
    if config["quantization"] != {"group_size": 64, "bits": 4}:
        raise ValueError("Expected group-64 INT4 quantization")
    return manifest
