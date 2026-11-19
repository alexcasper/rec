"""Torch runtime for the ``rocm`` backend (rec-9na).

Implements the engine-facing surface of ``native_runtime.NativeRuntime``
(make_session / prefill / decode_step / ``_state``) on the torch model layer
in ``rec_mlx.torch_rocm``. The model runs on a torch device (ROCm builds of
torch expose HIP GPUs as ``cuda``); prefill/decode_step return fp32 torch
logits and sampling happens on-device in the engine's torch loop
(``_generate_tokens_torch`` + :class:`TorchSampler`) — no mlx arrays and no
per-step host copies. Grammar-constrained decode and prefix reuse are not
supported, same as the cpu backend.

Device selection: ``REC_TORCH_DEVICE`` (default ``cuda``). ``cpu`` is
accepted so the full engine path is testable on any Linux box.
Activation/KV dtype: ``REC_TORCH_DTYPE`` = ``bf16`` (default) or ``fp32``
(conformance: isolates engine/device bugs from bf16 rounding).
"""

from __future__ import annotations

import os

import torch

from .backends import BackendError
from .torch_rocm import load_model


class TorchDeviceProfile:
    """Duck-typed stand-in for kernels.device.DeviceProfile."""

    workers = 0
    mlp_workers = 0
    head_workers = 0
    prefill_chunk_size = 2048

    def __init__(self, device: torch.device):
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device)
            hip = getattr(torch.version, "hip", None)
            self.chip = f"{name} (hip {hip})" if hip else name
        else:
            self.chip = "torch-cpu"


def resolve_device(requested: str | None = None) -> torch.device:
    name = requested or os.environ.get("REC_TORCH_DEVICE", "cuda")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BackendError(
            "The rocm backend needs a torch GPU device (ROCm torch exposes HIP as 'cuda'); "
            "none is available. Set REC_TORCH_DEVICE=cpu to run the torch path on CPU."
        )
    return device


_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_dtype(requested: str | None = None) -> torch.dtype:
    name = requested or os.environ.get("REC_TORCH_DTYPE", "bf16")
    if name not in _DTYPES:
        raise BackendError(f"REC_TORCH_DTYPE must be one of {', '.join(_DTYPES)}, got {name!r}")
    return _DTYPES[name]


def _sampler_logprobs(flat: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """mlx_lm-mirroring filter: log-softmax, then top-k and top-p masking.

    Mirrors ``mlx_lm.sample_utils`` on log-probabilities: top-k keeps the k
    highest and masks the rest to -inf; top-p keeps tokens whose
    ascending-order cumulative probability exceeds ``1 - top_p`` (boundary
    token included). The filtered *distribution* is identical to mlx; the
    draws are not (different RNG streams) — that is by design and asserted in
    ``tests/test_torch_sampling.py``.
    """
    lp = torch.log_softmax(flat.float(), dim=-1)
    vocab = lp.shape[-1]
    if 0 < top_k < vocab:
        # Exactly k survivors even on ties (mlx_lm argpartition keeps exactly k).
        keep = torch.topk(lp, top_k, dim=-1).indices
        keep_mask = torch.zeros_like(lp, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=keep, src=torch.ones_like(keep, dtype=torch.bool))
        lp = torch.where(keep_mask, lp, lp.new_full((), float("-inf")))
    if 0 < top_p < 1.0:
        ascending = torch.argsort(lp, dim=-1, stable=True)
        probs = torch.exp(lp)
        cum = torch.cumsum(probs[ascending], dim=-1)[torch.argsort(ascending, dim=-1, stable=True)]
        lp = torch.where(cum > 1 - top_p, lp, lp.new_full((), float("-inf")))
    return lp


class TorchSampler:
    """Temperature/top-p/top-k sampling on the torch device (option 1: no mlx
    in the decode loop). temp == 0 is exact argmax (greedy is the
    cross-backend-exact path); temp > 0 matches mlx_lm's filtered distribution
    with per-request reproducible torch draws (seeded Generator)."""

    def __init__(self, temperature: float, top_p: float, top_k: int, device: torch.device, seed: int):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.generator = torch.Generator(device=device.type)
        self.generator.manual_seed(seed)

    def sample(self, flat: torch.Tensor) -> int:
        if self.temperature == 0:
            return int(flat.argmax())
        lp = _sampler_logprobs(flat, self.temperature, self.top_p, self.top_k)
        probs = torch.softmax(lp / self.temperature, dim=-1)
        if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 0.0:
            # Degenerate filter state (e.g. top-k mass < 1 - top_p masks every
            # token). mlx categorical is undefined here; argmax is the safe
            # interpretation.
            return int(flat.argmax())
        return int(torch.multinomial(probs, 1, generator=self.generator))


class TorchSession:
    def __init__(self, runtime: TorchRuntime, max_new_tokens: int):
        self.runtime = runtime
        self.maximum = max_new_tokens
        self.cached_tokens = 0
        self.common_prefix_tokens = 0
        self.prompt_cache = None
        self._kv = None

    @torch.inference_mode()
    def prefill(self, tokens, *, prefix=None, should_stop=lambda: False, interrupt_between_layers=False):
        if prefix is not None and getattr(prefix, "tokens", None):
            raise ValueError("The rocm backend does not support resident prompt-prefix reuse")
        limit = len(tokens) + self.maximum
        if limit > self.runtime.context_limit:
            raise ValueError("Decode context limit exhausted")
        model = self.runtime.model
        self._kv = model.new_cache()
        ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        logits = model.forward(ids, self._kv)[:, -1:]
        self._state = {
            "kv": self._kv,  # KVCache exposes .nbytes
            "capacity": limit,
            "limit": limit,
            "growth_count": 0,
            "growth_seconds": 0.0,
            "position": len(tokens),
        }
        return logits  # torch fp32 [1, 1, V]; the engine's torch loop samples on-device

    @torch.inference_mode()
    def decode_step(self, token):
        model = self.runtime.model
        ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        logits = model.forward(ids, self._kv)
        self._state["position"] += 1
        return logits


class TorchRuntime:
    """Engine-facing runtime for the rocm backend."""

    def __init__(self, directory, *, context_limit: int, device: str | None = None,
                 dtype: str | None = None):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        self.model = load_model(directory, device=str(self.device), dtype=self.dtype)
        self.context_limit = context_limit
        self.device_profile = TorchDeviceProfile(self.device)
        self.prefill_chunk_size = TorchDeviceProfile.prefill_chunk_size
        self.metadata = [[0] * 64]

    def make_session(self, *, max_new_tokens=512):
        return TorchSession(self, max_new_tokens)
