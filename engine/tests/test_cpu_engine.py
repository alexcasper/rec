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


def test_rocm_backend_requires_torch_gpu(tiny_checkpoint, monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a torch GPU is present; rocm backend would load")
    monkeypatch.setenv("REC_BACKEND", "rocm")
    monkeypatch.delenv("REC_TORCH_DEVICE", raising=False)
    from rec_mlx.backends import BackendError

    with pytest.raises(BackendError, match="torch GPU"):
        Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=4096)


def test_out_of_vocab_prompt_ids_are_rejected(engine):
    for ids in ([1, 5, 512], [1, -1, 5]):
        with pytest.raises(ValueError, match="Token ids must be in"):
            list(engine.generate_tokens(ids, Sampling(temperature=0, max_tokens=2)))


def test_synthetic_tokenizer_ids_fit_the_vocab(tiny_checkpoint, tmp_path):
    """Specials and chat-template output must stay < vocab_size and be deterministic."""
    from rec_mlx.engine import load_tokenizer
    from rec_mlx.synthetic import build_checkpoint

    tokenizer = load_tokenizer(tiny_checkpoint)
    assert len(tokenizer) == 512
    specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|user|>", "<|assistant|>"]
    assert max(tokenizer.convert_tokens_to_ids(specials)) < 512
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "tok7 tok8"}], add_generation_prompt=True)
    ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids  # BatchEncoding is not a dict
    assert max(ids) < 512
    again = build_checkpoint(tmp_path / "again", "tiny")
    assert (again / "tokenizer.json").read_bytes() == (tiny_checkpoint / "tokenizer.json").read_bytes()
