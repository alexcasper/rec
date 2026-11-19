"""rec-jow: the tokenizer/vocab guard rejects only ids that reach the model on
every request, tolerates extra plain tokenizer ids past the vocab, and the
repair tool fixes checkpoints written before 92f9bc2."""

import json
import platform
import shutil

import pytest

from rec_mlx.engine import Engine, Sampling

pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="metal auto-detection wins on Apple Silicon",
)


def _load(ckpt, **kw):
    return Engine(ckpt, ckpt / "manifest.json", context_limit=4096, **kw)


def _write_pre_92f9bc2_tokenizer(directory, vocab_size):
    """rec_mlx.synthetic._write_tokenizer as it was before 92f9bc2, verbatim: tok0..tok2
    collide with the specials, so the chat specials land past the vocab."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from rec_mlx.synthetic import CHAT_TOKENS, SPECIAL_TOKENS

    vocabulary = {f"tok{i}": i for i in range(vocab_size)}
    for index, token in enumerate(SPECIAL_TOKENS):
        vocabulary[token] = index
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|end_of_text|>",
                            bos_token="<|begin_of_text|>",
                            additional_special_tokens=CHAT_TOKENS).save_pretrained(directory)
    if not (directory / "special_tokens_map.json").is_file():
        (directory / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>",
                        "additional_special_tokens": CHAT_TOKENS}))


@pytest.fixture
def broken_checkpoint(tiny_checkpoint, tmp_path):
    ckpt = tmp_path / "broken"
    shutil.copytree(tiny_checkpoint, ckpt)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (ckpt / name).unlink()
    _write_pre_92f9bc2_tokenizer(ckpt, 512)
    manifest = json.loads((ckpt / "manifest.json").read_text())
    manifest["model_id"] = "synthetic-tiny-flint-seed0"
    (ckpt / "manifest.json").write_text(json.dumps(manifest))
    return ckpt


def test_pre_92f9bc2_tokenizer_is_rejected_with_repair_hint(broken_checkpoint):
    from rec_mlx.chat import load_tokenizer

    tok = load_tokenizer(broken_checkpoint)
    # Hash-order dependent (PYTHONHASHSEED): 515..517 ids; the shipped cpu-baseline-ckpt has 516.
    # The chat role markers land past the vocab in every ordering.
    assert 512 < len(tok) <= 517
    assert min(tok.convert_tokens_to_ids(["<|user|>", "<|assistant|>"])) >= 512
    with pytest.raises(ValueError, match="repair-tokenizer") as err:
        _load(broken_checkpoint)
    assert "<|user|>" in str(err.value) and "reach the model on every request" in str(err.value)


def test_repair_tokenizer_fixes_load_and_keeps_weights(broken_checkpoint, tmp_path):
    from rec_mlx.synthetic import repair_tokenizer

    fixed = repair_tokenizer(broken_checkpoint, tmp_path / "fixed")
    assert (fixed / "model-00001.safetensors").read_bytes() == \
        (broken_checkpoint / "model-00001.safetensors").read_bytes()
    assert (broken_checkpoint / "tokenizer.json").read_bytes() != (fixed / "tokenizer.json").read_bytes()
    engine = _load(fixed)
    ids = engine.prompt([{"role": "user", "content": "tok7 tok8"}])
    assert max(ids) < 512 and engine.tokenizer_extra_ids == 0
    events = list(engine.generate_tokens(ids, Sampling(temperature=0, max_tokens=4)))
    assert any("metrics" in e for e in events)


def test_repair_tokenizer_refuses_non_synthetic_and_existing_dst(broken_checkpoint, tiny_checkpoint, tmp_path):
    from rec_mlx.synthetic import repair_tokenizer

    with pytest.raises(ValueError, match="not a rec_mlx.synthetic"):
        repair_tokenizer(tiny_checkpoint, tmp_path / "x")  # model_id 'tiny-flint-test'
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        repair_tokenizer(broken_checkpoint, tmp_path / "taken")


def test_extra_plain_tokenizer_ids_are_tolerated_and_rejected_per_request(tiny_checkpoint, tmp_path):
    """Real tokenizers carry a few ids past the vocab: load, but never feed them."""
    from transformers import PreTrainedTokenizerFast

    ckpt = tmp_path / "extra"
    shutil.copytree(tiny_checkpoint, ckpt)
    tok = PreTrainedTokenizerFast.from_pretrained(ckpt)
    assert tok.add_tokens(["xtra_one", "xtra_two", "xtra_three"]) == 3  # plain, not special
    tok.save_pretrained(ckpt)
    engine = _load(ckpt)
    assert engine.tokenizer_extra_ids == 3
    ids = engine.prompt([{"role": "user", "content": "tok7 xtra_two"}])
    assert max(ids) >= 512
    with pytest.raises(ValueError, match="Token ids must be in"):
        list(engine.generate_tokens(ids, Sampling(temperature=0, max_tokens=2)))
    ok = engine.prompt([{"role": "user", "content": "tok7 tok8"}])
    assert any("metrics" in e for e in engine.generate_tokens(ok, Sampling(temperature=0, max_tokens=2)))


def test_extra_special_id_past_vocab_is_rejected(tiny_checkpoint, tmp_path):
    from transformers import PreTrainedTokenizerFast

    ckpt = tmp_path / "special"
    shutil.copytree(tiny_checkpoint, ckpt)
    tok = PreTrainedTokenizerFast.from_pretrained(ckpt)
    from rec_mlx.synthetic import CHAT_TOKENS

    tok.add_special_tokens({"additional_special_tokens": CHAT_TOKENS + ["<|tool|>"]})
    tok.save_pretrained(ckpt)
    with pytest.raises(ValueError, match="special: \\[512\\]"):
        _load(ckpt)


def test_config_eos_past_vocab_is_rejected(tiny_checkpoint, tmp_path):
    ckpt = tmp_path / "eos"
    shutil.copytree(tiny_checkpoint, ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    cfg["eos_token_id"] = [2, 600]
    (ckpt / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="config eos_token_id: \\[600\\]"):
        _load(ckpt)
