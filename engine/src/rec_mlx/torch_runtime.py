"""Torch runtime for the ``rocm`` backend (rec-9na).

Implements the engine-facing surface of ``native_runtime.NativeRuntime``
(make_session / prefill / decode_step / ``_state``) on the torch model layer
in ``rec_mlx.torch_rocm``. The model runs on a torch device (ROCm builds of
torch expose HIP GPUs as ``cuda``); logits are handed back to the engine as
fp32 ``mx.array`` so the shared sampling loop (``_generate_tokens_native``)
is unchanged. Grammar-constrained decode and prefix reuse are not supported,
same as the cpu backend.

Device selection: ``REC_TORCH_DEVICE`` (default ``cuda``). ``cpu`` is
accepted so the full engine path is testable on any Linux box.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
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


class _KVBytes:
    def __init__(self, kv):
        self._kv = kv

    @property
    def nbytes(self):
        return sum(k.nbytes + v.nbytes for k, v in (e for e in self._kv if e is not None))


def _to_mx(logits: torch.Tensor) -> mx.array:
    """[1, 1, V] fp32 torch logits -> mx.array (host copy; V floats per step)."""
    return mx.array(np.ascontiguousarray(logits.detach().float().cpu().numpy()))


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
            "kv": _KVBytes(self._kv),
            "capacity": limit,
            "limit": limit,
            "growth_count": 0,
            "growth_seconds": 0.0,
            "position": len(tokens),
        }
        return _to_mx(logits)

    @torch.inference_mode()
    def decode_step(self, token):
        model = self.runtime.model
        ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        logits = model.forward(ids, self._kv)
        self._state["position"] += 1
        return _to_mx(logits)


class TorchRuntime:
    """Engine-facing runtime for the rocm backend."""

    def __init__(self, directory, *, context_limit: int, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16):
        self.device = resolve_device(device)
        self.model = load_model(directory, device=str(self.device), dtype=dtype)
        self.context_limit = context_limit
        self.device_profile = TorchDeviceProfile(self.device)
        self.prefill_chunk_size = TorchDeviceProfile.prefill_chunk_size
        self.metadata = [[0] * 64]

    def make_session(self, *, max_new_tokens=512):
        return TorchSession(self, max_new_tokens)
