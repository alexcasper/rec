"""rec-aip step 2: the lm_head cache and the sync-free single-token MoE path are
bit-identical to the per-step / host-synced paths they replace. The torch layer
is also the conformance reference (engine/validation/megakernel_mid)."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec_mlx import torch_rocm as T  # noqa: E402


@pytest.fixture(scope="module", params=["production", "synthetic"])
def ckpt(request, tmp_path_factory):
    from rec_mlx.synthetic import build_checkpoint

    kw = {"fmt": "production"} if request.param == "production" else {}
    return build_checkpoint(tmp_path_factory.mktemp(request.param) / "tiny", "tiny", **kw)


def decode_logits(model, prompt, n):
    """Prefill, then n single-token greedy steps; every step's logits."""
    out = []
    with torch.inference_mode():
        kv = model.new_cache()
        lg = model.forward(torch.tensor([prompt]), kv)
        out.append(lg)
        tok = int(lg[0, -1].argmax())
        for _ in range(n):
            lg = model.forward(torch.tensor([[tok]]), kv)
            out.append(lg)
            tok = int(lg[0, -1].argmax())
    return out


def old_paths(model):
    model.q["lm_head"].cache_dense = False
    model.moe_sync_free_decode = False
    return model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("plen", [5, 530])  # 530 + decode crosses the 513 sliding window
def test_new_paths_bit_identical(ckpt, dtype, plen):
    prompt = [1] + [6 + (37 * i) % 400 for i in range(plen - 1)]
    new = decode_logits(T.load_model(ckpt, dtype=dtype), prompt, 12)
    old = decode_logits(old_paths(T.load_model(ckpt, dtype=dtype)), prompt, 12)
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert torch.equal(a, b)


def test_decode_does_not_sync_expert_set(ckpt, monkeypatch):
    model = T.load_model(ckpt, dtype=torch.bfloat16)
    with torch.inference_mode():
        kv = model.new_cache()
        model.forward(torch.tensor([[1, 6, 7]]), kv)  # prefill keeps the synced path
        monkeypatch.setattr(torch, "unique", lambda *a, **k: (_ for _ in ()).throw(AssertionError("synced")))
        model.forward(torch.tensor([[8]]), kv)


def test_lm_head_cache_identity_and_invalidation(ckpt):
    model = T.load_model(ckpt, dtype=torch.bfloat16)
    lm = model.q["lm_head"]
    assert lm.cache_dense
    a = lm.dense(torch.bfloat16)
    assert lm.dense(torch.bfloat16) is a  # cached, not recomputed
    assert torch.equal(a, T.dequant(lm.packed, lm.scales, lm.biases, lm.bits, torch.bfloat16))
    f = lm.dense(torch.float32)  # dtype is part of the key
    assert f.dtype == torch.float32 and lm.cached_nbytes() == f.nbytes
    lm.scales.mul_(2)  # in-place weight change bumps the version: cache must refresh
    b = lm.dense(torch.float32)
    assert b is not f and torch.equal(b, T.dequant(lm.packed, lm.scales, lm.biases, lm.bits, torch.float32))


def test_cache_can_be_disabled(ckpt):
    model = T.load_model(ckpt, dtype=torch.bfloat16, cache_lm_head=False)
    lm = model.q["lm_head"]
    assert lm.dense(torch.bfloat16) is not lm.dense(torch.bfloat16) and lm.cached_nbytes() == 0


def test_cache_works_for_models_loaded_under_inference_mode(ckpt):
    """torch_rocm_bench.py and the engine load under inference_mode: no version counter."""
    with torch.inference_mode():
        model = T.load_model(ckpt, dtype=torch.bfloat16)
        lm = model.q["lm_head"]
        a = lm.dense(torch.bfloat16)
        assert lm.dense(torch.bfloat16) is a
        assert torch.equal(a, T.dequant(lm.packed, lm.scales, lm.biases, lm.bits, torch.bfloat16))
        lm.invalidate()
        assert lm.dense(torch.bfloat16) is not a
    new = decode_logits(model, [1, 6, 7, 8], 6)
    with torch.inference_mode():
        ref = old_paths(T.load_model(ckpt, dtype=torch.bfloat16))
    for x, y in zip(new, decode_logits(ref, [1, 6, 7, 8], 6)):
        assert torch.equal(x, y)
