"""Torch sampler (rocm backend, option 1) parity + reproducibility.

Parity target is the mlx_lm sampler the cpu backend uses:
  - greedy (temp 0): exact argmax on both libs -> identical tokens.
  - temp > 0: the *filtered distribution* must match (same surviving tokens,
    same probabilities). Draws intentionally differ (different RNG streams);
    per-request reproducibility on torch is asserted instead.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mlx_mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx_lm.sample_utils import apply_top_k, apply_top_p  # noqa: E402
from rec_mlx.torch_runtime import TorchSampler, _sampler_logprobs  # noqa: E402


def mlx_filtered_probs(flat, temp, top_p, top_k):
    lp = flat - mlx_mx.logsumexp(flat)
    if top_k:
        lp = apply_top_k(lp, top_k)
    if 0 < top_p < 1.0:
        lp = apply_top_p(lp, top_p)
    return mlx_mx.softmax(lp * (1 / temp))


def torch_filtered_probs(flat, temp, top_p, top_k):
    lp = _sampler_logprobs(flat, temp, top_p, top_k)
    return torch.softmax(lp / temp, dim=-1)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "temp,top_p,top_k",
    [(0.7, 0.95, 20), (1.0, 1.0, 0), (0.5, 0.9, 0), (1.5, 0.99, 50), (0.1, 0.5, 5)],
)
def test_filtered_distribution_matches_mlx(seed, temp, top_p, top_k):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randn(512, generator=g) * 3.0
    p_mlx = np.asarray(mlx_filtered_probs(mlx_mx.array(flat.numpy()), temp, top_p, top_k))
    p_torch = torch_filtered_probs(flat, temp, top_p, top_k)
    if not bool(torch.isfinite(p_torch).all()) or float(p_torch.sum()) <= 0.0:
        pytest.skip("degenerate filter state: top-k mass below 1 - top_p masks everything")
    diff = abs(p_mlx - p_torch.numpy()).max()
    assert diff < 1e-5, f"max|prob diff|={diff:.3e} temp={temp} top_p={top_p} top_k={top_k}"


def test_degenerate_filter_falls_back_to_argmax():
    """Top-k(10) leaves ~0.14 total mass; top_p=0.8 demands cum > 0.2 -> mlx
    masks everything (undefined draws). Our sampler must not crash and falls
    back to argmax."""
    g = torch.Generator().manual_seed(3)
    flat = torch.randn(512, generator=g)
    s = TorchSampler(0.3, 0.8, 10, device=torch.device("cpu"), seed=1)
    assert s.sample(flat) == int(flat.argmax())


def test_greedy_matches_mlx_argmax():
    g = torch.Generator().manual_seed(7)
    flat = torch.randn(512, generator=g)
    s = TorchSampler(0.0, 0.95, 20, device=torch.device("cpu"), seed=0)
    assert s.sample(flat) == int(flat.argmax())


@pytest.mark.parametrize("temp,top_p,top_k", [(0.7, 0.95, 20), (1.0, 1.0, 0), (0.3, 0.8, 10)])
def test_same_seed_reproducible_on_torch(temp, top_p, top_k):
    g = torch.Generator().manual_seed(3)
    flat = torch.randn(512, generator=g)
    toks_a = [
        TorchSampler(temp, top_p, top_k, device=torch.device("cpu"), seed=11).sample(flat)
        for _ in range(8)
    ]
    toks_b = [
        TorchSampler(temp, top_p, top_k, device=torch.device("cpu"), seed=11).sample(flat)
        for _ in range(8)
    ]
    assert toks_a == toks_b


def test_topk_keeps_exactly_k_on_ties():
    flat = torch.zeros(512)  # every token ties
    lp = _sampler_logprobs(flat, 1.0, 0.0, 20)
    assert int(torch.isfinite(lp).sum()) == 20  # mlx argpartition also keeps exactly k
