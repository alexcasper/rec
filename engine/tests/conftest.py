"""Shared test fixtures and backend guards.

Test modules that exercise the resident Metal kernels (megakernel, KVStorage
banks, resident sampler) only run on the metal backend; they are skipped
automatically elsewhere so the Linux/CPU suite stays green.
"""


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


def build_tiny_checkpoint(directory):
    """Materialize a valid tiny Flint checkpoint (random weights, real loader path)."""
    from rec_mlx.synthetic import build_checkpoint

    return build_checkpoint(directory, "tiny", seed=0, model_id="tiny-flint-test")


@pytest.fixture
def tiny_checkpoint(tmp_path):
    return build_tiny_checkpoint(tmp_path / "ckpt")
