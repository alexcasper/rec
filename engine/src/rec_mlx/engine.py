import hashlib
import json
import logging
import resource
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .backends import METAL_MIN_MEMORY_BYTES, BackendError, detect_backend
from .chat import load_tokenizer, normalize_messages
from .checkpoint import validate
from .detokenize import StreamingText

logger = logging.getLogger(__name__)


class ContextLimitError(ValueError):
    pass


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 1024
    seed: int = 0
    presence_penalty: float = 0.0

    def __post_init__(self):
        for name in ("temperature", "top_p", "presence_penalty"):
            if type(getattr(self, name)) not in (int, float):
                raise ValueError(f"{name} must be a number")
        for name in ("top_k", "max_tokens", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be in [0, 2] and top_p in (0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if self.top_k < 0 or self.max_tokens < 1:
            raise ValueError("top_k must be nonnegative and max_tokens positive")
        if not 0 <= self.seed <= 2**64 - self.max_tokens:
            raise ValueError("seed plus max_tokens must fit in unsigned 64-bit random keys")


class Engine:
    def __init__(
        self,
        directory: Path,
        manifest_path: Path,
        *,
        context_limit: int,
        enforce_tool_schemas: bool = False,
        backend: str | None = None,
    ):
        import mlx.core as mx

        if backend is None:
            self.backend = detect_backend()
        else:
            from .backends import SUPPORTED_BACKENDS

            if backend not in SUPPORTED_BACKENDS:
                raise BackendError(
                    f"backend must be one of {', '.join(SUPPORTED_BACKENDS)}, got {backend!r}"
                )
            self.backend = detect_backend()
            if self.backend.name != backend:
                raise BackendError(f"The {backend} backend is not available on this machine (detected {self.backend.name})")
        is_metal = self.backend.name == "metal"
        if is_metal and self.backend.physical_memory_bytes < METAL_MIN_MEMORY_BYTES:
            raise RuntimeError("This build targets Macs with at least 48 GiB of memory")
        if not is_metal:
            logger.warning(
                "Running on the %s backend: resident-kernel decode, grammar-constrained "
                "sampling and prompt-prefix preparation are disabled.",
                self.backend.name,
            )
        memory = self.backend.physical_memory_bytes
        device = mx.device_info()
        profile = None
        self.identity = validate(directory, manifest_path)
        config = json.loads((directory / "config.json").read_text())
        from .quantized import QATModel
        if self.identity["max_context"] != QATModel.context_limit:
            raise ValueError("Manifest context limit does not match the engine capacity")
        if not 1 <= context_limit <= QATModel.context_limit:
            raise ValueError(f"Context limit must be between 1 and {QATModel.context_limit} tokens")
        self.context_limit = context_limit
        self._prompt_cache = None
        self._prefix_preparation = None
        self.mx = mx
        self.tokenizer = load_tokenizer(directory)
        self.vocab_size = config["vocab_size"]
        self.tokenizer_extra_ids = self._check_tokenizer_vocab(config)
        self._constraints = None
        self.enforce_tool_schemas = enforce_tool_schemas
        self._constraints_lock = threading.Lock()
        self.eos = set(config["eos_token_id"])
        if not self.eos:
            raise ValueError("Checkpoint has no EOS tokens")
        start = time.perf_counter()
        mx.set_cache_limit(0)
        mx.clear_cache()
        mx.reset_peak_memory()
        if is_metal:
            from .kernels import build_decoder

            self.runtime, self.resident_sampler = build_decoder(QATModel(directory), optimized=True)
            # Packing replaces native weights with lazy views. Materialize them on their creation
            # stream before HTTP threads use the native prefill model.
            mx.eval(self.runtime.model.native.parameters())
            # Retain enough allocator cache to reuse the first 1.78-GiB KV block.
            cache_limit = 2 * 1024**3
            mx.set_cache_limit(cache_limit)
        elif self.backend.name == "cpu":
            from .native_runtime import NativeRuntime

            self.resident_sampler = None
            self.runtime = NativeRuntime(QATModel(directory), context_limit=context_limit)
            cache_limit = 0
        elif self.backend.name == "rocm":
            from .torch_runtime import TorchRuntime

            self.resident_sampler = None
            self.runtime = TorchRuntime(directory, context_limit=context_limit)
            cache_limit = 0
        else:
            raise BackendError(
                f"The {self.backend.name} backend is not implemented yet; see docs/amd-port-assessment.md"
            )
        self.startup = {
            "device": device,
            "chat_template": "flint",
            "chat_template_sha256": hashlib.sha256(
                self.tokenizer.chat_template.encode()
            ).hexdigest(),
            "allocation_cache_limit_bytes": cache_limit,
            "load_seconds": time.perf_counter() - start,
            "load_peak_mlx_bytes": mx.get_peak_memory(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "physical_memory_bytes": memory,
            "device_profile": profile.chip if profile is not None else (
                self.runtime.device_profile.chip if self.backend.name == "rocm" else "CPU"
            ),
            "backend": self.backend.name,
            "weight_format": self.identity["weight_format"],
            "kv_format": self.identity["kv_format"],
        }

    # Probe for the ids the chat template emits on its own (role markers etc.).
    _PROBE_MESSAGES = [
        {"role": "system", "content": "probe"},
        {"role": "user", "content": "probe"},
        {"role": "assistant", "content": "probe"},
        {"role": "user", "content": "probe"},
    ]

    def _check_tokenizer_vocab(self, config) -> int:
        """Reject a tokenizer whose *structural* ids fall outside the model vocab.

        Every request carries the special tokens (BOS/EOS/chat role markers) that
        the chat template emits, and generation stops on the config EOS ids. Any of
        them >= vocab_size is an out-of-bounds embedding gather on every request
        (a GPU hardware fault on rocm, rec-ju9), so that fails at load. Extra
        *plain* tokenizer ids past the vocab are tolerated (real tokenizers carry a
        few): user text that maps onto one is rejected per request by the range
        check in _generate_tokens. Returns the number of tolerated extra ids.
        """
        vocab = self.vocab_size
        structural = {"special": set(self.tokenizer.all_special_ids),
                      "config eos_token_id": set(config["eos_token_id"])}
        try:
            structural["chat template"] = set(self.prompt(self._PROBE_MESSAGES))
        except Exception as exc:  # a template that cannot render fails later, loudly, per request
            logger.warning("Chat template probe failed at load: %s", exc)
        bad = {where: sorted(i for i in ids if not 0 <= i < vocab) for where, ids in structural.items()}
        bad = {where: ids for where, ids in bad.items() if ids}
        if bad:
            names = sorted({self.tokenizer.convert_ids_to_tokens(i) for ids in bad.values() for i in ids
                            if i >= 0}, key=str)
            raise ValueError(
                f"Tokenizer ids outside the model vocab [0, {vocab}) reach the model on every request: "
                + "; ".join(f"{where}: {ids}" for where, ids in bad.items())
                + f" ({', '.join(map(str, names))}). Checkpoints written before 92f9bc2 by "
                "rec_mlx.synthetic have this tokenizer bug: rewrite the tokenizer with "
                "`python -m rec_mlx.synthetic --repair-tokenizer SRC --out DST` (weights unchanged)."
            )
        extra = max(len(self.tokenizer) - vocab, 0)
        if extra:
            logger.warning(
                "Tokenizer has %d ids past the model vocab %d (none structural); requests that "
                "encode to them are rejected", extra, vocab)
        return extra

    @property
    def constraints(self):
        with self._constraints_lock:
            if self._constraints is None:
                from .grammar import ConstraintCompiler

                self._constraints = ConstraintCompiler(
                    self.tokenizer, enforce_tool_schemas=self.enforce_tool_schemas
                )
            return self._constraints

    def prompt(self, messages: list[dict], *, tools=None) -> list[int]:
        return self.tokenizer.apply_chat_template(
            normalize_messages(messages),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

    def generate(self, messages: list[dict], sampling: Sampling, *, grammar=None):
        events = self.generate_tokens(self.prompt(messages), sampling, grammar=grammar)
        try:
            for event in events:
                if "metrics" in event:
                    yield event
                elif event["text"]:
                    yield {"text": event["text"]}
        finally:
            events.close()

    def generate_tokens(
        self,
        ids,
        sampling,
        *,
        should_stop=lambda: False,
        grammar=None,
        return_logprobs=False,
    ):
        # Metal streams are thread-local; HTTP handlers run outside the model-loading thread.
        # The CPU backend has no streams: run directly on the default device.
        if self.backend.name == "metal":
            stream_context = self.mx.stream(self.mx.new_stream(self.mx.gpu))
        else:
            stream_context = nullcontext()
        with stream_context:
            from .prefix import PrefillCancelled

            try:
                with grammar.session() if grammar is not None else nullcontext(None) as constraint:
                    yield from self._generate_tokens(
                        ids,
                        sampling,
                        should_stop=should_stop,
                        grammar=constraint,
                        return_logprobs=return_logprobs,
                    )
            except PrefillCancelled:
                self._sync()
                self.mx.clear_cache()

    def prepare_prefix(self, *, should_stop=lambda: False):
        if self.backend.name != "metal":
            return None  # prompt-prefix preparation is a resident-kernel feature
        from .prefix import PrefillCancelled

        pending = self._prefix_preparation
        if pending is None:
            return None
        mx = self.mx
        chunk = self.runtime.prefill_chunk_size
        origin = len(pending.origin.tokens)
        end = origin + (len(pending.tokens) - origin) // chunk * chunk
        start = time.perf_counter()
        prepared = 0
        with mx.stream(mx.new_stream(mx.gpu)):
            while (origin if pending.cache is None else len(pending.cache.tokens)) < end:
                if should_stop():
                    break
                prefix = pending.origin if pending.cache is None else pending.cache
                target = len(prefix.tokens) + chunk
                session = self.runtime.make_session(max_new_tokens=0)
                try:
                    session.prefill(
                        pending.tokens[:target],
                        prefix=prefix.fork(),
                        should_stop=should_stop,
                        interrupt_between_layers=True,
                    )
                except PrefillCancelled:
                    # A partial chunk wrote past the committed prefix in shared global KV; never reuse it.
                    if len(session.prompt_cache.tokens) != target:
                        break
                pending.cache = session.prompt_cache
                prepared += chunk
            mx.synchronize()
        return {"prepared_tokens": prepared, "preparation_seconds": time.perf_counter() - start}

    def _schedule_prefix(self, ids, generated):
        if self.backend.name != "metal":
            return
        from .prefix import PrefixPreparation

        if len(generated) >= self.runtime.prefill_chunk_size:
            self._prefix_preparation = PrefixPreparation(self._prompt_cache, ids + generated)

    def _take_prefix(self, ids):
        prefix, self._prompt_cache = self._prompt_cache, None
        pending, self._prefix_preparation = self._prefix_preparation, None
        if pending is not None and pending.cache is not None:
            prepared = pending.cache
            if ids[: len(prepared.tokens)] != prepared.tokens:
                from .prefix import select_prefix

                prepared, _ = select_prefix(prepared, ids)
            if prepared is not None and len(prepared.tokens) > len(pending.origin.tokens):
                prefix = prepared
        return prefix

    def _sync(self):
        if self.backend.name == "metal":
            self.mx.synchronize()

    def _generate_tokens(self, ids, sampling, *, should_stop, grammar=None, return_logprobs=False):
        if len(ids) + sampling.max_tokens > self.context_limit:
            raise ContextLimitError(
                "Prompt plus output allowance exceeds the configured context limit"
            )
        # Shared precondition for every backend: an id >= vocab_size is an
        # out-of-bounds embedding gather -- a GPU hardware fault on rocm that
        # kills the process, silent garbage on MLX.
        if ids and not (0 <= min(ids) and max(ids) < self.vocab_size):
            raise ValueError(f"Token ids must be in [0, {self.vocab_size}); got {min(ids)}..{max(ids)}")
        if self.backend.name == "rocm":
            yield from self._generate_tokens_torch(
                ids, sampling, should_stop=should_stop, grammar=grammar, return_logprobs=return_logprobs
            )
            return
        if self.backend.name != "metal":
            yield from self._generate_tokens_native(
                ids, sampling, should_stop=should_stop, grammar=grammar, return_logprobs=return_logprobs
            )
            return
        mx = self.mx
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
        )
        mx.reset_peak_memory()
        session = self.runtime.make_session(max_new_tokens=sampling.max_tokens)
        unfiltered_t1 = (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, 0)
        mx.random.seed(sampling.seed)
        stops = self.eos if grammar is None else set(grammar.stops)
        generated = []
        token_logprobs = []
        start = time.perf_counter()
        reason = "length"
        try:
            preparation = self._prefix_preparation
            prefix = self._take_prefix(ids)
            from .prefix import PrefillCancelled

            try:
                logits = session.prefill(ids, prefix=prefix, should_stop=should_stop)
            except PrefillCancelled:
                if preparation is not None and prefix is preparation.cache:
                    # Keep the original input anchor until a new request completes prefill.
                    self._prompt_cache = preparation.origin
                    preparation.cache = session.prompt_cache
                    self._prefix_preparation = preparation
                else:
                    self._prompt_cache = session.prompt_cache
                raise
            del prefix, preparation
            self._prompt_cache = session.prompt_cache
            text_stream = StreamingText(self.tokenizer)
            if sampling.presence_penalty:
                session._state["presence_mask"] = mx.zeros(logits.size, mx.float32)
            mx.eval(logits)
            if self.runtime.device_profile.chip == "M4 Max":
                from .kernels.filtered_sampling import make_sampler

                sampler = make_sampler(
                    temp=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                )
            prefill_end = time.perf_counter()
            prefill_peak = mx.get_peak_memory()
            first = None
            packet = []
            computed_tokens = 0
            for index in range(sampling.max_tokens):
                if should_stop():
                    reason = "stop"
                    break
                if index:
                    if not packet:
                        count = min(8, sampling.max_tokens - index)
                        options = {"logprobs": token_logprobs} if return_logprobs else {}
                        packet = self.resident_sampler.block(
                            session._state,
                            generated[-1],
                            count,
                            sampling.seed + index,
                            temperature=sampling.temperature,
                            top_p=sampling.top_p,
                            top_k=sampling.top_k,
                            presence_penalty=sampling.presence_penalty,
                            grammar=grammar,
                            **options,
                        )
                        computed_tokens += len(packet)
                    token = packet.pop(0)
                else:
                    if not bool(mx.all(mx.isfinite(logits)).item()):
                        raise FloatingPointError("The inference engine produced nonfinite logits")
                    flat = logits.reshape(-1).astype(mx.float32)
                    if grammar is not None:
                        flat = grammar.first_logits(flat)
                    if sampling.temperature == 0:
                        token = int(mx.argmax(flat).item())
                    elif unfiltered_t1:
                        # Match the upstream first-token random key exactly.
                        token = int(
                            mx.random.categorical(flat, key=mx.random.key(sampling.seed)).item()
                        )
                    else:
                        token = int(sampler(flat - mx.logsumexp(flat)).item())
                    if grammar is not None:
                        grammar.accept_first(token, session._state)
                    if return_logprobs:
                        report = flat / sampling.temperature if sampling.temperature else flat
                        token_logprobs.append(float((report[token] - mx.logsumexp(report)).item()))
                    computed_tokens += 1
                    if sampling.presence_penalty:
                        session._state["presence_mask"][token] = 1.0
                if first is None:
                    first = time.perf_counter()
                generated.append(token)
                details = {"logprob": token_logprobs[index]} if return_logprobs else {}
                if token in stops:
                    reason = "stop"
                    delta = text_stream.finish(generated[:-1])
                    yield {"token": token, "text": delta, "finish_reason": reason, **details}
                    break
                delta = text_stream.add_token(token)
                if index + 1 == sampling.max_tokens:
                    delta += text_stream.finish(generated)
                yield {
                    "token": token,
                    "text": delta,
                    "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
                    **details,
                }
            mx.synchronize()
            end = time.perf_counter()
            decode_seconds = end - first if first is not None else 0.0
            sliding_bytes = sum(
                cache.nbytes
                for row, cache in zip(
                    self.runtime.metadata[:49], self._prompt_cache.layers, strict=True
                )
                if not row[1]
            )
            yield {
                "finish_reason": reason,
                "metrics": {
                    "prompt_tokens": len(ids),
                    "cached_prompt_tokens": session.cached_tokens,
                    "common_prefix_tokens": session.common_prefix_tokens,
                    "prefix_checkpoint_bytes": self._prompt_cache.checkpoint_bytes,
                    "prefilled_tokens": len(ids) - session.cached_tokens,
                    "prompt_cache_bytes": session._state["kv"].nbytes
                    + sliding_bytes
                    + self._prompt_cache.checkpoint_bytes,
                    "native_sliding_cache_bytes": sliding_bytes,
                    "shared_kv_bytes": session._state["kv"].nbytes,
                    "decode_kv_capacity_tokens": session._state["capacity"],
                    "decode_kv_growth_count": session._state["growth_count"],
                    "decode_kv_growth_seconds": session._state["growth_seconds"],
                    "generated_tokens": len(generated),
                    "token_ids": generated,
                    **(
                        {"token_logprobs": token_logprobs[: len(generated)]}
                        if return_logprobs
                        else {}
                    ),
                    "ttft_seconds": first - start if first is not None else None,
                    "prefill_seconds": prefill_end - start,
                    "decode_seconds": decode_seconds,
                    "request_seconds": end - start,
                    "decode_tokens_per_second": max(0, len(generated) - 1) / decode_seconds
                    if decode_seconds
                    else 0.0,
                    "prefill_peak_mlx_bytes": prefill_peak,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "computed_tokens": computed_tokens,
                    "decode_strategy": "resident_sampling",
                    **(grammar.metrics() if grammar is not None else {}),
                },
            }
        finally:
            if generated:
                self._schedule_prefix(ids, generated)
            self._sync()
            del session
            mx.clear_cache()

    def _generate_tokens_torch(self, ids, sampling, *, should_stop, grammar=None, return_logprobs=False):
        """Torch-native generation for the rocm backend: sampling happens on
        the model's torch device via :class:`TorchSampler`. No mlx arrays and
        no per-step device->host->mlx copies (option 1, rec-9na). The event
        and metrics contract is identical to ``_generate_tokens_native``;
        memory-accounting metrics come from the torch allocator (0 on
        non-CUDA devices). Filtered-sampling draws are per-request
        reproducible on torch (seeded generator) but intentionally not
        identical to mlx draws; greedy (temp 0) is exact on every backend.
        """
        if grammar is not None:
            raise NotImplementedError(
                "Grammar-constrained decoding requires the resident sampler (metal backend)"
            )
        import torch

        from .torch_runtime import TorchSampler

        device = self.runtime.device
        sampler = TorchSampler(
            sampling.temperature, sampling.top_p, sampling.top_k, device=device, seed=sampling.seed
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def peak_bytes() -> int:
            return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

        session = self.runtime.make_session(max_new_tokens=sampling.max_tokens)
        stops = self.eos
        generated = []
        token_logprobs = []
        seen = set()
        start = time.perf_counter()
        reason = "length"
        try:
            logits = session.prefill(ids, prefix=None, should_stop=should_stop)
            self._prompt_cache = session.prompt_cache
            text_stream = StreamingText(self.tokenizer)
            prefill_end = time.perf_counter()
            prefill_peak = peak_bytes()
            first = None
            for index in range(sampling.max_tokens):
                if should_stop():
                    reason = "stop"
                    break
                flat = logits.reshape(-1).float()
                if not bool(torch.isfinite(flat).all()):
                    raise FloatingPointError("The inference engine produced nonfinite logits")
                if sampling.presence_penalty and seen:
                    seen_idx = torch.tensor(sorted(seen), dtype=torch.long, device=flat.device)
                    flat[seen_idx] -= sampling.presence_penalty
                token = sampler.sample(flat)
                if return_logprobs:
                    report = flat / sampling.temperature if sampling.temperature else flat
                    token_logprobs.append(float((report[token] - torch.logsumexp(report, 0)).item()))
                if first is None:
                    first = time.perf_counter()
                generated.append(token)
                seen.add(token)
                details = {"logprob": token_logprobs[index]} if return_logprobs else {}
                if token in stops:
                    reason = "stop"
                    delta = text_stream.finish(generated[:-1])
                    yield {"token": token, "text": delta, "finish_reason": reason, **details}
                    break
                delta = text_stream.add_token(token)
                if index + 1 == sampling.max_tokens:
                    delta += text_stream.finish(generated)
                yield {
                    "token": token,
                    "text": delta,
                    "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
                    **details,
                }
                logits = session.decode_step(token)
            self._sync()
            end = time.perf_counter()
            decode_seconds = end - first if first is not None else 0.0
            peak = peak_bytes()
            yield {
                "finish_reason": reason,
                "metrics": {
                    "prompt_tokens": len(ids),
                    "cached_prompt_tokens": session.cached_tokens,
                    "common_prefix_tokens": session.common_prefix_tokens,
                    "prefix_checkpoint_bytes": 0,
                    "prefilled_tokens": len(ids) - session.cached_tokens,
                    "prompt_cache_bytes": session._state["kv"].nbytes,
                    "native_sliding_cache_bytes": 0,
                    "shared_kv_bytes": session._state["kv"].nbytes,
                    "decode_kv_capacity_tokens": session._state["capacity"],
                    "decode_kv_growth_count": 0,
                    "decode_kv_growth_seconds": 0.0,
                    "generated_tokens": len(generated),
                    "token_ids": generated,
                    **(
                        {"token_logprobs": token_logprobs[: len(generated)]}
                        if return_logprobs
                        else {}
                    ),
                    "ttft_seconds": first - start if first is not None else None,
                    "prefill_seconds": prefill_end - start,
                    "decode_seconds": decode_seconds,
                    "request_seconds": end - start,
                    "decode_tokens_per_second": max(0, len(generated) - 1) / decode_seconds
                    if decode_seconds
                    else 0.0,
                    # Schema-stable keys; the values are torch-allocator peaks.
                    "prefill_peak_mlx_bytes": prefill_peak,
                    "peak_mlx_bytes": peak,
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "computed_tokens": len(generated),
                    "decode_strategy": "torch",
                },
            }
        finally:
            self._sync()
            del session

    def _generate_tokens_native(self, ids, sampling, *, should_stop, grammar=None, return_logprobs=False):
        """Pure-MLX generation used by the cpu backend (no resident kernel)."""
        if grammar is not None:
            raise NotImplementedError(
                "Grammar-constrained decoding requires the resident sampler (metal backend)"
            )
        mx = self.mx
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
        )
        mx.reset_peak_memory()
        session = self.runtime.make_session(max_new_tokens=sampling.max_tokens)
        mx.random.seed(sampling.seed)
        stops = self.eos
        generated = []
        token_logprobs = []
        seen = set()
        start = time.perf_counter()
        reason = "length"
        try:
            logits = session.prefill(ids, prefix=None, should_stop=should_stop)
            mx.eval(logits)
            text_stream = StreamingText(self.tokenizer)
            prefill_end = time.perf_counter()
            prefill_peak = mx.get_peak_memory()
            first = None
            for index in range(sampling.max_tokens):
                if should_stop():
                    reason = "stop"
                    break
                if not bool(mx.all(mx.isfinite(logits)).item()):
                    raise FloatingPointError("The inference engine produced nonfinite logits")
                flat = logits.reshape(-1).astype(mx.float32)
                if sampling.presence_penalty and seen:
                    for token_id in seen:
                        flat[token_id] -= sampling.presence_penalty
                if sampling.temperature == 0:
                    token = int(mx.argmax(flat).item())
                else:
                    token = int(sampler(flat - mx.logsumexp(flat)).item())
                if return_logprobs:
                    report = flat / sampling.temperature if sampling.temperature else flat
                    token_logprobs.append(float((report[token] - mx.logsumexp(report)).item()))
                if first is None:
                    first = time.perf_counter()
                generated.append(token)
                seen.add(token)
                details = {"logprob": token_logprobs[index]} if return_logprobs else {}
                if token in stops:
                    reason = "stop"
                    delta = text_stream.finish(generated[:-1])
                    yield {"token": token, "text": delta, "finish_reason": reason, **details}
                    break
                delta = text_stream.add_token(token)
                if index + 1 == sampling.max_tokens:
                    delta += text_stream.finish(generated)
                yield {
                    "token": token,
                    "text": delta,
                    "finish_reason": "length" if index + 1 == sampling.max_tokens else None,
                    **details,
                }
                logits = session.decode_step(token)
                mx.eval(logits)
            self._sync()
            end = time.perf_counter()
            decode_seconds = end - first if first is not None else 0.0
            yield {
                "finish_reason": reason,
                "metrics": {
                    "prompt_tokens": len(ids),
                    "cached_prompt_tokens": session.cached_tokens,
                    "common_prefix_tokens": session.common_prefix_tokens,
                    "prefix_checkpoint_bytes": 0,
                    "prefilled_tokens": len(ids) - session.cached_tokens,
                    "prompt_cache_bytes": session._state["kv"].nbytes,
                    "native_sliding_cache_bytes": 0,
                    "shared_kv_bytes": session._state["kv"].nbytes,
                    "decode_kv_capacity_tokens": session._state["capacity"],
                    "decode_kv_growth_count": 0,
                    "decode_kv_growth_seconds": 0.0,
                    "generated_tokens": len(generated),
                    "token_ids": generated,
                    **(
                        {"token_logprobs": token_logprobs[: len(generated)]}
                        if return_logprobs
                        else {}
                    ),
                    "ttft_seconds": first - start if first is not None else None,
                    "prefill_seconds": prefill_end - start,
                    "decode_seconds": decode_seconds,
                    "request_seconds": end - start,
                    "decode_tokens_per_second": max(0, len(generated) - 1) / decode_seconds
                    if decode_seconds
                    else 0.0,
                    "prefill_peak_mlx_bytes": prefill_peak,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "computed_tokens": len(generated),
                    "decode_strategy": "torch" if self.backend.name == "rocm" else "native_mlx",
                },
            }
        finally:
            self._sync()
            del session
            mx.clear_cache()
