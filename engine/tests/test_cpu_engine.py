"""End-to-end Linux/CPU engine path: load, prefill, decode, metrics, guards."""

import platform

import pytest

from rec_mlx.backends import detect_backend
from rec_mlx.engine import ContextLimitError, Engine, Sampling

pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="metal auto-detection wins on Apple Silicon",
)


@pytest.fixture
def engine(tiny_checkpoint):
    return Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=4096)


def test_engine_runs_native_generation(engine):
    assert engine.backend.name == "cpu"
    assert engine.startup["backend"] == "cpu"
    assert engine.startup["device_profile"] == "CPU"
    events = list(engine.generate_tokens([1, 5, 6, 7, 8], Sampling(temperature=0, max_tokens=8)))
    assert events[-1]["finish_reason"] == "length"
    metrics = events[-1]["metrics"]
    assert metrics["decode_strategy"] == "native_mlx"
    assert len(metrics["token_ids"]) == 8
    assert metrics["generated_tokens"] == 8
    for event in events[:-1]:
        assert "token" in event
        assert "text" in event


def test_generation_is_deterministic_greedy(engine):
    def run():
        return [
            event["token"]
            for event in engine.generate_tokens([1, 5, 6, 7, 8], Sampling(temperature=0, max_tokens=5))
            if "token" in event
        ]

    assert run() == run()


def test_backend_selection_rejects_mismatch(tiny_checkpoint):
    from rec_mlx.backends import BackendError

    detected = detect_backend().name
    wrong = "metal" if detected != "metal" else "cpu"
    with pytest.raises(BackendError):
        Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=4096, backend=wrong)


def test_grammar_is_metal_only(engine):
    from types import SimpleNamespace

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *error):
            return False

    fake_grammar = SimpleNamespace(session=lambda: _Ctx())
    events = engine.generate_tokens([1, 5, 6], Sampling(temperature=0, max_tokens=2), grammar=fake_grammar)
    with pytest.raises(NotImplementedError):
        for _ in events:
            pass


def test_context_limit_still_enforced(tiny_checkpoint):
    engine = Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=64)
    with pytest.raises(ContextLimitError):
        list(engine.generate_tokens(list(range(1, 60)), Sampling(temperature=0, max_tokens=16)))


def test_rocm_backend_selectable_but_unimplemented(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv("REC_BACKEND", "rocm")
    from rec_mlx.backends import BackendError

    with pytest.raises(BackendError):
        Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=4096)
