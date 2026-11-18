"""rec-9na: torch model layer (rocm backend) vs the production MLX model.

Runs on CPU torch so the Linux suite exercises the rocm code path without a
GPU; GPU conformance is tests/torch_rocm_tiny_check.py on the device host.
"""

import platform

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rec_mlx.engine import Engine, Sampling  # noqa: E402
from rec_mlx.quantized import QATModel  # noqa: E402
from rec_mlx.torch_rocm import load_model  # noqa: E402

PROMPT = [1, 5, 6, 7, 8, 9, 10, 11]


def production_format(src, dst):
    """Rewrite a synthetic checkpoint the way RecursiveSI/flint-int4 ships:
    U8 packed weights, BF16 scales/biases/norms/router, no expert_bias."""
    import shutil

    shutil.copytree(src, dst)
    shard = dst / "model-00001.safetensors"
    out = {}
    for name, value in mx.load(str(shard)).items():
        if name.endswith(".mlp.gate.expert_bias"):
            continue
        if value.dtype == mx.uint32:
            out[name] = value.view(mx.uint8)
        elif value.dtype == mx.float32:
            out[name] = value.astype(mx.bfloat16)
        else:
            out[name] = value
    mx.eval(out)  # mx.load is lazy: materialize before overwriting the same file
    mx.save_safetensors(str(shard), out)
    return dst


def mlx_logits(ckpt, tokens):
    native = QATModel(ckpt).native
    return np.array(native(mx.array([tokens])).astype(mx.float32))[0]


@torch.no_grad()
def torch_logits(ckpt, tokens, dtype=torch.float32):
    return load_model(ckpt, device="cpu", dtype=dtype).forward(torch.tensor([tokens])).numpy()[0]


def test_dequant_matches_mx_dequantize(tiny_checkpoint):
    model = load_model(tiny_checkpoint, device="cpu", dtype=torch.float32)
    weights = mx.load(str(tiny_checkpoint / "model-00001.safetensors"))
    config = QATModel(tiny_checkpoint).config["mlx_export"]
    checked = 0
    for base, ql in [*model.q.items(), ("model.embed_tokens", model.embed)]:
        bits = 8 if base == "model.embed_tokens" else (
            config["routed_expert_bits"] if ".switch_mlp." in base else config["bits"])
        ref = mx.dequantize(weights[base + ".weight"], weights[base + ".scales"],
                            weights[base + ".biases"], group_size=64, bits=bits)
        np.testing.assert_array_equal(ql.dense(torch.float32).numpy(), np.array(ref), err_msg=base)
        checked += 1
    assert checked == len(model.q) + 1 >= 10


def test_fp32_logits_match_mlx_model(tiny_checkpoint):
    ref, ours = mlx_logits(tiny_checkpoint, PROMPT), torch_logits(tiny_checkpoint, PROMPT)
    assert np.abs(ref - ours).max() < 1e-4 * max(1.0, np.abs(ref).max())
    assert (ref.argmax(-1) == ours.argmax(-1)).all()


def test_sliding_window_beyond_513_matches_mlx(tiny_checkpoint):
    tokens = [int(t) for t in np.random.default_rng(0).integers(3, 512, size=600)]
    ref, ours = mlx_logits(tiny_checkpoint, tokens), torch_logits(tiny_checkpoint, tokens)
    np.testing.assert_allclose(ours[-80:], ref[-80:], atol=1e-4 * max(1.0, np.abs(ref).max()))


@torch.no_grad()
def test_kv_cache_incremental_matches_full(tiny_checkpoint):
    model = load_model(tiny_checkpoint, device="cpu", dtype=torch.float32)
    ids = torch.tensor([PROMPT * 3])
    full = model.forward(ids)
    kv = model.new_cache()
    steps = [model.forward(ids[:, :10], kv)]
    steps += [model.forward(ids[:, t:t + 1], kv) for t in range(10, ids.shape[1])]
    assert (torch.cat(steps, 1) - full).abs().max() < 1e-4
    assert all(k.dtype == torch.float32 for k, _ in kv)
    bf = load_model(tiny_checkpoint, device="cpu", dtype=torch.bfloat16)
    kv = bf.new_cache()
    bf.forward(ids[:, :4], kv)
    assert all(k.dtype == v.dtype == torch.bfloat16 for k, v in kv)  # bf16 KV cache


def test_production_format_checkpoint_loads_and_matches(tiny_checkpoint, tmp_path):
    ckpt = production_format(tiny_checkpoint, tmp_path / "prod")
    model = load_model(ckpt, device="cpu", dtype=torch.bfloat16)
    assert model.q["lm_head"].scales.dtype == torch.bfloat16  # kept in checkpoint dtype
    assert model.t["model.layers.1.mlp.gate.expert_bias"].abs().max() == 0  # filled like sanitize()
    ref = mlx_logits(ckpt, PROMPT)
    ours = torch_logits(ckpt, PROMPT, dtype=torch.bfloat16)
    truth = torch_logits(ckpt, PROMPT, dtype=torch.float32)  # == MLX fp32 to ~1e-6 (test above)
    scale = max(1.0, np.abs(truth).max())
    # bf16 torch must stay within 3% of fp32 truth and be at least as faithful
    # as production MLX bf16 (measured: torch rms 4.9e-3, MLX rms 1.9e-2)
    err_ours, err_mlx = np.abs(ours - truth).max(), np.abs(ref - truth).max()
    assert err_ours < 0.03 * scale, err_ours
    assert err_ours <= err_mlx * 1.05, (err_ours, err_mlx)


def test_zero_router_rejected_by_torch_loader(tiny_checkpoint):
    shard = tiny_checkpoint / "model-00001.safetensors"
    weights = mx.load(str(shard))
    weights["model.layers.1.mlp.gate.weight"] = mx.zeros_like(weights["model.layers.1.mlp.gate.weight"])
    mx.eval(weights)  # mx.load is lazy: materialize before overwriting the same file
    mx.save_safetensors(str(shard), weights)
    with pytest.raises(ValueError, match="rec-9as"):
        load_model(tiny_checkpoint, device="cpu")


@pytest.mark.skipif(platform.system() == "Darwin" and platform.machine() == "arm64",
                    reason="metal auto-detection wins on Apple Silicon")
def test_engine_rocm_backend_on_torch_cpu_matches_cpu_backend(tiny_checkpoint, monkeypatch):
    def greedy(engine):
        events = list(engine.generate_tokens(PROMPT, Sampling(temperature=0, max_tokens=12)))
        return events[-1]["metrics"]

    cpu = greedy(Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=4096))
    monkeypatch.setenv("REC_BACKEND", "rocm")
    monkeypatch.setenv("REC_TORCH_DEVICE", "cpu")
    engine = Engine(tiny_checkpoint, tiny_checkpoint / "manifest.json", context_limit=4096)
    assert engine.startup["backend"] == "rocm" and engine.startup["device_profile"] == "torch-cpu"
    rocm = greedy(engine)
    assert rocm["decode_strategy"] == "torch"
    assert rocm["generated_tokens"] == 12
    # torch runs bf16 activations, MLX-cpu fp32 here: allow a near-tie flip
    # only after a common prefix of >= 8 tokens
    common = next((i for i, (a, b) in enumerate(zip(cpu["token_ids"], rocm["token_ids"])) if a != b), 12)
    assert common >= 8, (cpu["token_ids"], rocm["token_ids"])
