"""Pure-MLX runtime for the ``cpu`` backend.

Reimplements the subset of the Metal resident runtime's surface that
``rec_mlx.engine`` consumes (make_session / prefill / forward) using the
portable native model with mlx-lm KV caches. No Metal kernels, no KVStorage
banks, no resident sampler, no grammar-constrained decode, no prefix
preparation: decode samples token-by-token through the engine's own loop.
"""


import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache


class CPUDeviceProfile:
    """Duck-typed stand-in for kernels.device.DeviceProfile."""

    chip = "CPU"
    workers = 0
    mlp_workers = 0
    head_workers = 0
    prefill_chunk_size = 2048


class NativeSession:
    def __init__(self, runtime, max_new_tokens):
        self.runtime = runtime
        self.maximum = max_new_tokens
        self.cached_tokens = 0
        self.common_prefix_tokens = 0
        self._cache = None

    def prefill(self, tokens, *, prefix=None, should_stop=lambda: False, interrupt_between_layers=False):
        if prefix is not None and getattr(prefix, "tokens", None):
            raise ValueError("The cpu backend does not support resident prompt-prefix reuse")
        model = self.runtime.model.native
        limit = len(tokens) + self.maximum
        if limit > self.runtime.context_limit:
            raise ValueError("Decode context limit exhausted")
        self._cache = [
            KVCache() if kind == "full_attention" else RotatingKVCache(max_size=self.runtime.model.config["sliding_window"], keep=0)
            for kind in model.model.layer_types
        ]
        hidden = model.model(mx.array([tokens], dtype=mx.int32), self._cache)
        logits = model.lm_head(hidden[:, -1:])
        mx.eval(logits, [entry.state for entry in self._cache])
        self.prompt_cache = None  # prefix preparation is Metal-only for now
        self._state = {
            "kv": _CacheBytes(self._cache),
            "capacity": limit,
            "limit": limit,
            "growth_count": 0,
            "growth_seconds": 0.0,
            "position": len(tokens),
        }
        self._tokens = tokens
        return logits

    def decode_step(self, token):
        model = self.runtime.model.native
        hidden = model.model(mx.array([[token]], dtype=mx.int32), self._cache)
        logits = model.lm_head(hidden[:, -1:])
        self._state["position"] += 1
        return logits


class _CacheBytes:
    """Reports the KV footprint of a session's mlx-lm caches."""

    def __init__(self, caches):
        self._caches = caches

    @property
    def nbytes(self):
        return sum(cache.state[0].nbytes + cache.state[1].nbytes for cache in self._caches)


class NativeRuntime:
    """Engine-facing runtime for the cpu backend."""

    def __init__(self, model, *, context_limit):
        self.model = model
        self.context_limit = context_limit
        self.device_profile = CPUDeviceProfile()
        self.prefill_chunk_size = CPUDeviceProfile.prefill_chunk_size
        # Metal metrics zip runtime.metadata against prompt-cache layers; the
        # CPU backend reports no resident-kernel metadata.
        self.metadata = [[0] * 64]
        mx.eval(self.model.native.parameters())

    def make_session(self, *, max_new_tokens=512):
        return NativeSession(self, max_new_tokens)
