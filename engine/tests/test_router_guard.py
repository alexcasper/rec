"""rec-9as: degenerate MoE routers are rejected at load time."""

import mlx.core as mx
import numpy as np
import pytest

from rec_mlx.checkpoint import assert_router_nondegenerate
from rec_mlx.quantized import QATModel
from rec_mlx.synthetic import PRESETS

CONFIG = {"num_hidden_layers": 3, "mlp_only_layers": [0]}
NAME = "model.layers.{}.mlp.gate.weight"


def test_guard_rejects_all_zero_router():
    with pytest.raises(ValueError, match="all-zero"):
        assert_router_nondegenerate({NAME.format(1): np.zeros((8, 16))}, CONFIG)


def test_guard_rejects_identical_rows():
    w = np.tile(np.random.default_rng(0).normal(size=(1, 16)), (8, 1))
    with pytest.raises(ValueError, match="identical rows"):
        assert_router_nondegenerate({NAME.format(2): w}, CONFIG)


def test_guard_accepts_seeded_router_and_ignores_dense_layers():
    rng = np.random.default_rng(0)
    assert_router_nondegenerate(
        {NAME.format(0): np.zeros((8, 16)),  # dense layer: not a router, ignored
         NAME.format(1): rng.normal(size=(8, 16)),
         NAME.format(2): rng.normal(size=(8, 16))},
        CONFIG,
    )


def test_qat_model_refuses_zero_router_checkpoint(tiny_checkpoint):
    shard = tiny_checkpoint / "model-00001.safetensors"
    weights = mx.load(str(shard))
    name = NAME.format(1)
    weights[name] = mx.zeros_like(weights[name])
    mx.save_safetensors(str(shard), weights)
    with pytest.raises(ValueError, match="rec-9as"):
        QATModel(tiny_checkpoint)


def test_synthetic_tiny_checkpoint_loads(tiny_checkpoint):
    weights = mx.load(str(tiny_checkpoint / "model-00001.safetensors"))
    w = np.array(weights[NAME.format(1)].astype(mx.float32))
    assert np.abs(w).max() > 0 and len({tuple(r) for r in w.round(6)}) == w.shape[0]
    QATModel(tiny_checkpoint)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_are_flint_shaped(preset):
    c = PRESETS[preset]
    assert c["sliding_window"] == 513
    assert c["mlx_export"] == {"bits": 4, "routed_expert_bits": 4, "embedding_bits": 8}
    assert len(c["layer_types"]) == c["num_hidden_layers"]
    assert c["num_experts"] > c["num_experts_per_tok"] >= 1
