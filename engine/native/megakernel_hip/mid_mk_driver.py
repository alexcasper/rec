"""Pack a mid-8L Flint safetensors checkpoint (zero-copy U8/U32 INT4/INT8 + BF16/F32
scales/biases per docs/rocm-model-layer.md s2), run the persistent multi-workgroup
`mid_generate` megakernel (`flint_mid.so`), validate token-for-token and last-step
logits against a reference, and benchmark decode tok/s per run-definitions.md.

Usage (on ROCm host, torch-free process for kernel launch):
    # 1. Validate correctness (generates mid_mk_io.npz):
    python mid_mk_driver.py --ckpt ~/rec-ve/ckpt/cpu-baseline-mid \
                            --prompt=1,5,6,7 --n-out=16 --workers=16 \
                            --out=runs/mid_mk_io.npz

    # 2. Check against numpy reference (no torch/MLX needed) or torch reference:
    python mid_mk_driver.py --verify=runs/mid_mk_io.npz --ckpt ~/rec-ve/ckpt/cpu-baseline-mid

    # 3. Benchmark decode32 tok/s after 512-token prefill (per run-definitions.md):
    python mid_mk_driver.py --bench --ckpt ~/rec-ve/ckpt/cpu-baseline-mid --workers=16
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MID_NL = 8
MID_HID = 2048
MID_HEADS = 32
MID_NKV = 4
MID_HDIM = 128
MID_QDIM = MID_HEADS * MID_HDIM
MID_KVDIM = MID_NKV * MID_HDIM
MID_DENSE_I = 2048
MID_MOE_I = 512
MID_SH_I = 1024
MID_NEXP = 32
MID_TOPK = 2
MID_VOCAB = 150144
MID_WIN = 513
EPS = 1e-6
SCALE = 1.0 / np.sqrt(MID_HDIM)
THETA = 10000.0
TIE_TOL = 5e-2
LOGIT_RTOL = 2e-2

_ST_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "U32": np.uint32,
    "I32": np.int32,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
}


def f32_to_bf16(v: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def load_safetensors_raw(ckpt_dir: Path) -> dict[str, tuple[np.ndarray, str]]:
    out: dict[str, tuple[np.ndarray, str]] = {}
    for p in sorted(Path(ckpt_dir).glob("model-*.safetensors")):
        with open(p, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
            base = 8 + n
            header.pop("__metadata__", None)
            for name, info in header.items():
                dt = _ST_DTYPES[info["dtype"]]
                start, end = info["data_offsets"]
                f.seek(base + start)
                buf = f.read(end - start)
                arr = np.frombuffer(buf, dtype=dt).reshape(info["shape"]).copy()
                out[name] = (arr, info["dtype"])
    return out


def to_bf16_bits(arr: np.ndarray, dtype_tag: str) -> np.ndarray:
    if dtype_tag == "BF16":
        return np.ascontiguousarray(arr, dtype=np.uint16)
    return f32_to_bf16(arr.astype(np.float32))


def pack_mid_bank(ckpt_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, str]]]:
    """Pack a mid-8L checkpoint into a contiguous byte bank + 140-entry offset table.

    Zero-copy for packed U8/U32 weights (little-endian U32 bytes == U8 layout per
    docs/rocm-model-layer.md s1-s2); scales/biases/norms stored as BF16 uint16 bits.
    """
    raw = load_safetensors_raw(ckpt_dir)
    bank = bytearray()
    offsets: list[int] = []

    def align2():
        if len(bank) & 1:
            bank.append(0)

    def append_bf16(name: str, fallback_zeros: int = 0):
        align2()
        offsets.append(len(bank))
        if name in raw:
            arr, dt = raw[name]
            bits = to_bf16_bits(arr, dt)
        else:
            bits = np.zeros(fallback_zeros, dtype=np.uint16)
        bank.extend(bits.tobytes())

    def append_qlinear(base: str):
        align2()
        offsets.append(len(bank))
        w_arr, _ = raw[base + ".weight"]
        s_arr, s_dt = raw[base + ".scales"]
        b_arr, b_dt = raw[base + ".biases"]
        # Zero-copy raw packed bytes (U8 or U32 little-endian are byte-identical)
        w_bytes = np.ascontiguousarray(w_arr).view(np.uint8)
        s_bits = to_bf16_bits(s_arr, s_dt)
        b_bits = to_bf16_bits(b_arr, b_dt)
        bank.extend(w_bytes.tobytes())
        bank.extend(s_bits.tobytes())
        bank.extend(b_bits.tobytes())

    # 0..55: Per-layer dense bf16 tensors (7 per layer)
    for l in range(MID_NL):
        pre = f"model.layers.{l}"
        append_bf16(f"{pre}.input_layernorm.weight")
        append_bf16(f"{pre}.post_attention_layernorm.weight")
        append_bf16(f"{pre}.self_attn.q_norm.weight")
        append_bf16(f"{pre}.self_attn.k_norm.weight")
        append_bf16(f"{pre}.self_attn.softmax_offset")
        if l == 0:
            append_bf16("__none__", 0)
            append_bf16("__none__", 0)
        else:
            append_bf16(f"{pre}.mlp.gate.weight")
            append_bf16(f"{pre}.mlp.gate.expert_bias", fallback_zeros=MID_NEXP)

    # 56: Final norm
    append_bf16("model.norm.weight")

    # 57: Embed tokens (INT8 packed + bf16 s + bf16 b)
    append_qlinear("model.embed_tokens")

    # 58..137: Per-layer Q4 tensors (10 per layer)
    for l in range(MID_NL):
        ap = f"model.layers.{l}.self_attn"
        mp = f"model.layers.{l}.mlp"
        append_qlinear(f"{ap}.q_proj")
        append_qlinear(f"{ap}.k_proj")
        append_qlinear(f"{ap}.v_proj")
        append_qlinear(f"{ap}.o_proj")
        if l == 0:
            append_qlinear(f"{mp}.gate_proj")
            append_qlinear(f"{mp}.up_proj")
            append_qlinear(f"{mp}.down_proj")
            for _ in range(3):
                align2()
                offsets.append(len(bank))
        else:
            append_qlinear(f"{mp}.switch_mlp.gate_proj")
            append_qlinear(f"{mp}.switch_mlp.up_proj")
            append_qlinear(f"{mp}.switch_mlp.down_proj")
            append_qlinear(f"{mp}.shared_expert.gate_proj")
            append_qlinear(f"{mp}.shared_expert.up_proj")
            append_qlinear(f"{mp}.shared_expert.down_proj")

    # 138: LM head
    append_qlinear("lm_head")

    # 139: Total bank byte length sentinel
    align2()
    offsets.append(len(bank))
    assert len(offsets) == 140, f"Expected 140 offsets, got {len(offsets)}"

    return np.frombuffer(bytes(bank), dtype=np.uint8).copy(), np.array(offsets, dtype=np.int64), raw


def dequant_np(raw: dict[str, tuple[np.ndarray, str]], base: str, bits: int = 4, idx=None) -> np.ndarray:
    """Dequantize a QLinear tensor (or slice `idx`) using BF16-rounded scales/biases
    (matching the kernel's BF16 s/b storage) to float64."""
    w_arr, _ = raw[base + ".weight"]
    s_arr, s_dt = raw[base + ".scales"]
    b_arr, b_dt = raw[base + ".biases"]
    if idx is not None:
        w_arr = w_arr[idx]
        s_arr = s_arr[idx]
        b_arr = b_arr[idx]
    s = bf16_to_f32(to_bf16_bits(s_arr, s_dt)).astype(np.float64)
    b = bf16_to_f32(to_bf16_bits(b_arr, b_dt)).astype(np.float64)
    w_u8 = np.ascontiguousarray(w_arr).view(np.uint8)
    if bits == 8:
        q = w_u8.reshape(*s.shape[:-1], -1).astype(np.float64)
    else:
        low = w_u8 & 0x0F
        high = (w_u8 >> 4) & 0x0F
        q = np.stack([low, high], axis=-1).reshape(*s.shape[:-1], -1).astype(np.float64)
    s_rep = np.repeat(s, 64, axis=-1)
    b_rep = np.repeat(b, 64, axis=-1)
    return q * s_rep + b_rep


def rms_np(x: np.ndarray, w_bf: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w_bf


def silu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.5 * x))


def rope_np(t: np.ndarray, positions: np.ndarray) -> np.ndarray:
    inv = THETA ** (-2.0 * np.arange(MID_HDIM // 2, dtype=np.float64) / MID_HDIM)
    th = positions[:, None].astype(np.float64) * inv[None, :]
    c, s = np.cos(th)[:, None, :], np.sin(th)[:, None, :]
    a, b = t[..., :MID_HDIM // 2], t[..., MID_HDIM // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def forward_mid_np(
    raw: dict[str, tuple[np.ndarray, str]], seq: list[int], return_experts: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Portable fp64 numpy reference forward for mid-8L over `seq` (returns [T, V] fp32 logits).
    When `return_experts=True`, also returns `step_experts` int32 [T, MID_NL, MID_TOPK]
    (layer 0 filled with -1; layers 1..7 hold the router top-k expert indices in rank order).
    Mirrors tiny_mk_np_reference.py / rec_mlx.model.Model semantics exactly."""
    T = len(seq)
    seq_arr = np.array(seq, dtype=np.int64)
    x = dequant_np(raw, "model.embed_tokens", bits=8, idx=seq_arr)
    step_experts = np.full((T, MID_NL, MID_TOPK), -1, dtype=np.int32) if return_experts else None

    def get_bf(name: str, fallback_zeros: int = 0) -> np.ndarray:
        if name in raw:
            arr, dt = raw[name]
            return bf16_to_f32(to_bf16_bits(arr, dt)).astype(np.float64)
        return np.zeros(fallback_zeros, dtype=np.float64)

    positions = np.arange(T, dtype=np.int64)
    for l in range(MID_NL):
        pre = f"model.layers.{l}"
        ap = f"{pre}.self_attn"
        mp = f"{pre}.mlp"
        is_sliding = (l % 2 == 0)

        h = rms_np(x, get_bf(f"{pre}.input_layernorm.weight"))
        wq = dequant_np(raw, f"{ap}.q_proj")
        wk = dequant_np(raw, f"{ap}.k_proj")
        wv = dequant_np(raw, f"{ap}.v_proj")
        wo = dequant_np(raw, f"{ap}.o_proj")

        q = rms_np((h @ wq.T).reshape(T, MID_HEADS, MID_HDIM), get_bf(f"{ap}.q_norm.weight"))
        k = rms_np((h @ wk.T).reshape(T, MID_NKV, MID_HDIM), get_bf(f"{ap}.k_norm.weight"))
        v = (h @ wv.T).reshape(T, MID_NKV, MID_HDIM)

        if is_sliding:
            q = rope_np(q, positions)
            k = rope_np(k, positions)

        sinks = get_bf(f"{ap}.softmax_offset")
        att = np.zeros((T, MID_HEADS, MID_HDIM), dtype=np.float64)
        qpos = positions[:, None]
        kpos = positions[None, :]
        mask = kpos > qpos
        if is_sliding:
            mask = mask | (kpos < qpos - (MID_WIN - 1))

        for head in range(MID_HEADS):
            kv = head // (MID_HEADS // MID_NKV)
            sc = (q[:, head] @ k[:, kv].T) * SCALE
            sc = np.where(mask, -np.inf, sc)
            sc_with_sink = np.concatenate([sc, np.full((T, 1), sinks[head], dtype=np.float64)], axis=1)
            mx_sc = np.max(sc_with_sink, axis=1, keepdims=True)
            pr = np.exp(sc_with_sink - mx_sc)
            pr = pr / np.sum(pr, axis=1, keepdims=True)
            att[:, head] = pr[:, :T] @ v[:, kv]

        x = x + att.reshape(T, -1) @ wo.T
        hn = rms_np(x, get_bf(f"{pre}.post_attention_layernorm.weight"))

        if l == 0:
            wg = dequant_np(raw, f"{mp}.gate_proj")
            wu = dequant_np(raw, f"{mp}.up_proj")
            wd = dequant_np(raw, f"{mp}.down_proj")
            g = hn @ wg.T
            u = hn @ wu.T
            x = x + (silu_np(g) * u) @ wd.T
        else:
            rw = get_bf(f"{mp}.gate.weight").reshape(MID_NEXP, MID_HID)
            rb = get_bf(f"{mp}.gate.expert_bias", fallback_zeros=MID_NEXP)
            scores = 1.0 / (1.0 + np.exp(-(hn @ rw.T)))
            top = np.argsort(-(scores + rb[None, :]), axis=1, kind="stable")[:, :MID_TOPK]
            if step_experts is not None:
                step_experts[:, l, :] = top.astype(np.int32)
            mix = np.take_along_axis(scores, top, axis=1)
            mix = mix / (np.sum(mix, axis=1, keepdims=True) + 1e-20)

            uniq_e = sorted(set(int(e) for e in top.reshape(-1)))
            exp_g = {e: dequant_np(raw, f"{mp}.switch_mlp.gate_proj", idx=e) for e in uniq_e}
            exp_u = {e: dequant_np(raw, f"{mp}.switch_mlp.up_proj", idx=e) for e in uniq_e}
            exp_d = {e: dequant_np(raw, f"{mp}.switch_mlp.down_proj", idx=e) for e in uniq_e}

            moe = np.zeros_like(x)
            for t_idx in range(T):
                for j in range(MID_TOPK):
                    e = int(top[t_idx, j])
                    moe[t_idx] += mix[t_idx, j] * (exp_d[e] @ (silu_np(exp_g[e] @ hn[t_idx]) * (exp_u[e] @ hn[t_idx])))

            w_sg = dequant_np(raw, f"{mp}.shared_expert.gate_proj")
            w_su = dequant_np(raw, f"{mp}.shared_expert.up_proj")
            w_sd = dequant_np(raw, f"{mp}.shared_expert.down_proj")
            sg = hn @ w_sg.T
            su = hn @ w_su.T
            x = x + moe + (silu_np(sg) * su) @ w_sd.T

    fn = rms_np(x, get_bf("model.norm.weight"))
    w_lm = dequant_np(raw, "lm_head")
    out_logits = (fn @ w_lm.T).astype(np.float32)
    if step_experts is not None:
        return out_logits, step_experts
    return out_logits


def verify_run(npz_path: Path, ckpt_dir: Path) -> bool:
    d = np.load(npz_path)
    raw = load_safetensors_raw(ckpt_dir)
    prompt = d["prompt"].tolist()
    kt = d["tokens"].tolist()
    klg = d["last_logits"]

    # Single causal pass over full trajectory seq = prompt + kt
    full_seq = list(prompt) + list(kt)
    all_logits = forward_mid_np(raw, full_seq)
    base = len(prompt) - 1

    ref_tokens, bad, ties = [], [], 0
    for step, ktok in enumerate(kt):
        lg = all_logits[base + step]
        want = int(lg.argmax())
        ref_tokens.append(want)
        margin = float(lg.max() - lg[ktok])
        if want != ktok:
            if margin > TIE_TOL:
                bad.append((step, want, ktok, margin))
            else:
                ties += 1

    # Last-step logits correspond to all_logits[-1] (at pos = len(prompt) + len(kt) - 1)
    ref_last = all_logits[-1]
    rel = float(np.abs(klg - ref_last).max() / max(float(np.abs(ref_last).max()), 1e-6))
    ok = (not bad) and (rel < LOGIT_RTOL)

    print("kernel tokens:", kt)
    print("ref    tokens:", ref_tokens, "(teacher-forced)")
    for step, want, got, margin in bad:
        print(f"  step {step}: ref {want} kernel {got} margin {margin:.3e} > {TIE_TOL}")
    if ties:
        print(f"near-ties accepted: {ties} (margin <= {TIE_TOL})")
    print(f"last-step logits rel maxabs err {rel:.3e} (tol {LOGIT_RTOL})")
    print("MATCH" if ok else "MISMATCH")
    return ok


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(HERE / "flint_mid.so"))
    lib.mid_query_occupancy.restype = ctypes.c_int
    lib.mid_query_occupancy.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.mid_session_create.restype = ctypes.c_void_p
    lib.mid_session_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mid_session_run.restype = ctypes.c_int
    lib.mid_session_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.mid_session_run_ex.restype = ctypes.c_int
    lib.mid_session_run_ex.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.mid_session_run_ex2.restype = ctypes.c_int
    lib.mid_session_run_ex2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.mid_session_free.restype = None
    lib.mid_session_free.argtypes = [ctypes.c_void_p]
    return lib


def query_occupancy(lib: ctypes.CDLL, mode: int = 0) -> dict[str, int]:
    cus = ctypes.c_int(0)
    b_per_cu = ctypes.c_int(0)
    max_active = ctypes.c_int(0)
    wave = ctypes.c_int(0)
    rc = lib.mid_query_occupancy(
        ctypes.c_int(mode),
        ctypes.byref(cus),
        ctypes.byref(b_per_cu),
        ctypes.byref(max_active),
        ctypes.byref(wave),
    )
    assert rc == 0, f"mid_query_occupancy failed with {rc}"
    return {
        "cus": cus.value,
        "max_blocks_per_cu": b_per_cu.value,
        "max_co_resident_workers": max_active.value,
        "wave_size": wave.value,
        "compiled_hard_cap": 64,
    }


def shard_sha256(ckpt_dir: Path) -> str:
    h = hashlib.sha256()
    with open(ckpt_dir / "model-00001.safetensors", "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def case_matrix(vocab: int = MID_VOCAB) -> dict[str, dict]:
    """Fixed case matrix matching engine/validation/megakernel_mid/mid_mk_check.py."""
    def tok(n: int) -> int:
        return 6 + n % (vocab - 6)

    rng = np.random.default_rng(0)
    wrap = [1] + [int(t) for t in rng.integers(6, vocab, size=511)]
    return {
        "short4": {"prompt": [1, tok(37), tok(1000), tok(52000)], "n_out": 24},
        "chat": {"prompt": [1, 4, tok(777), tok(8888), 5], "n_out": 24},
        "wrap512": {"prompt": wrap, "n_out": 24},
    }


def assert_valid_launch_outputs(
    rc: int,
    ctrl: np.ndarray,
    out_tokens: np.ndarray,
    last_logits: np.ndarray | None = None,
    step_logits: np.ndarray | None = None,
    all_experts: np.ndarray | None = None,
) -> None:
    """Reject failed, aborted, or no-op/unwritten launches (rec-tan.5 / R2 / R5)."""
    if rc != 0 or int(ctrl[2]) != 0 or int(ctrl[1]) == 0:
        raise RuntimeError(
            f"mid_session_run_ex2 failed: rc={rc}, control={ctrl.tolist()} "
            f"(phase={int(ctrl[1])}, abort={int(ctrl[2])}; check HSA_OVERRIDE_GFX_VERSION is unset for gfx1150)"
        )
    if len(out_tokens) > 0 and not np.any(out_tokens != 0):
        raise RuntimeError("mid_session_run_ex2 produced all-zero out_tokens (unwritten output buffer)")
    if last_logits is not None and not np.any(last_logits != 0):
        raise RuntimeError("mid_session_run_ex2 produced all-zero last_logits (unwritten output buffer)")
    if step_logits is not None and not np.any(step_logits != 0):
        raise RuntimeError("mid_session_run_ex2 produced all-zero step_logits (unwritten output buffer)")
    if all_experts is not None and bool((all_experts[:, 1:, :] < 0).any()):
        raise RuntimeError("mid_session_run_ex2 left unwritten -1 entries in MoE layers 1..7 of step_experts")


def emit_conformance_suite(ckpt_dir: Path, out_dir: Path, workers: int = 16,
                           builds: tuple[str, ...] = ("fp32kv", "bf16kv", "fp32_bf16kv"),
                           emit_experts: bool = False, only_cases: tuple[str, ...] | None = None) -> None:
    """Emit rec-mk-out/1 npz files for cases (short4, chat, wrap512) across requested builds:
      - fp32kv      : mode=2 (`mid_generate_fp32kv`, kv_dtype='fp32', pure fp32 compute + fp32 KV)
      - bf16kv      : mode=0 (`mid_generate`, kv_dtype='bf16', production c264e52 bf16 build)
      - fp32_bf16kv : mode=1 (`mid_generate_fp32_bf16kv`, kv_dtype='bf16', pure fp32 compute + bf16 KV)
    When `emit_experts=True`, passes an int32 [P + n_out, MID_NL, MID_TOPK] side buffer to
    `mid_session_run_ex2` and saves:
      - `chosen_experts`: int32 [n_out, MID_NL, MID_TOPK] (decode steps 0..n_out-1, at pos P-1..P+n_out-2)
      - `step_experts`  : int32 [P + n_out, MID_NL, MID_TOPK] (all positions 0..P+n_out-1, per README 4a)
    When `emit_experts=False` (default), passes nullptr so expert emission costs zero device work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = shard_sha256(ckpt_dir)
    print(f"ckpt: {ckpt_dir} | sha256: {sha} | emit_experts={emit_experts}")
    bank, off, _ = pack_mid_bank(ckpt_dir)
    cases = case_matrix(MID_VOCAB)
    if only_cases:  # e.g. ('short4', 'chat'): skip wrap512's multi-second launch on display GPUs (rec-6ob)
        unknown = set(only_cases) - set(cases)
        if unknown:
            raise SystemExit(f"unknown --cases {sorted(unknown)}; known {sorted(cases)}")
        cases = {k: v for k, v in cases.items() if k in only_cases}

    lib = load_lib()
    max_total = 600
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(max_total))

    build_specs = {
        "fp32kv": (2, "fp32", "flint_mid_megakernel.hip:mid_generate_fp32kv (fp32-compute, fp32-KV)"),
        "bf16kv": (0, "bf16", "flint_mid_megakernel.hip:mid_generate (production bf16-KV build)"),
        "fp32_bf16kv": (1, "bf16", "flint_mid_megakernel.hip:mid_generate_fp32_bf16kv (fp32-compute, bf16-KV)"),
    }

    for bname in builds:
        mode, kv_dtype, impl_str = build_specs[bname]
        for cname, cspec in cases.items():
            prompt_arr = np.array(cspec["prompt"], dtype=np.int32)
            p_len = len(prompt_arr)
            n_out = int(cspec["n_out"])
            out_tokens = np.zeros(n_out, dtype=np.int32)
            step_logits = np.zeros((n_out, MID_VOCAB), dtype=np.float32)
            last_logits = np.zeros(MID_VOCAB, dtype=np.float32)
            all_experts = (
                np.full((p_len + n_out, MID_NL, MID_TOPK), -1, dtype=np.int32)
                if emit_experts
                else None
            )

            ctrl = np.zeros(4, dtype=np.uint32)
            t0 = time.perf_counter()
            rc = lib.mid_session_run_ex2(
                sess,
                prompt_arr.ctypes.data,
                ctypes.c_int(p_len),
                out_tokens.ctypes.data,
                ctypes.c_int(n_out),
                step_logits.ctypes.data,
                last_logits.ctypes.data,
                ctypes.c_int(workers),
                ctypes.c_int(mode),
                ctrl.ctypes.data,
                all_experts.ctypes.data if all_experts is not None else None,
            )
            dt = time.perf_counter() - t0
            assert_valid_launch_outputs(rc, ctrl, out_tokens, last_logits, step_logits, all_experts)
            out_path = out_dir / f"{cname}_{bname}.npz"
            payload = dict(
                schema=np.array("rec-mk-out/1"),
                case=np.array(cname),
                ckpt_sha256=np.array(sha),
                impl=np.array(f"{impl_str} workers={workers}"),
                kv_dtype=np.array(kv_dtype),
                prompt=prompt_arr,
                tokens=out_tokens,
                step_logits=step_logits,
                last_logits=last_logits,
            )
            if all_experts is not None:
                # Decode step i (0 <= i < n_out) corresponds to forward position P - 1 + i
                payload["chosen_experts"] = all_experts[p_len - 1 : p_len - 1 + n_out]
                payload["step_experts"] = all_experts
            np.savez(out_path, **payload)
            print(
                f"wrote {out_path.name:26s} | {dt*1e3:6.1f} ms | kv={kv_dtype:4s} | "
                f"tokens[:6]={out_tokens[:6].tolist()}"
                + (f" | chosen_experts={payload['chosen_experts'].shape}" if all_experts is not None else "")
            )

    lib.mid_session_free(sess)


def _probe_cotenants() -> list[str]:
    """Return active GPU co-tenant processes (e.g. llama-server on :8080/:8090/:8091)."""
    import subprocess
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,pcpu,pmem,args"], text=True)
    except Exception:
        return []
    tenants = []
    for line in out.splitlines():
        if any(k in line for k in ("llama-server", "rec-mlx")) and "grep" not in line:
            tenants.append(line.strip())
    return tenants


def run_benchmark(ckpt_dir: Path, requested_workers: int = 64, mode: int = 0) -> None:
    """Execute the corrected decode32 differential benchmark protocol (R1-R8).

    - Starts with `requested_workers` (default 64, hard cap 64 per R3); queries
      `hipOccupancyMaxActiveBlocksPerMultiprocessor` (R6) and probes whether
      `requested_workers` completes without barrier abort (`control[2] == 0`, R2/R5).
      Also runs at occupancy-clamped / CU-count workers when `requested_workers`
      exceeds co-resident capacity or aborts.
    - Uses alternating ABBA pair order over 4 pairs (`(A,B)`, `(B,A)`, `(A,B)`, `(B,A)`)
      per R4 to cancel linear thermal/clock drift on shared iGPUs.
    - Verifies row-validity conditions on every launch (R2/R5):
        1. `rc == 0` and `control[2] == 0` (no barrier spin-timeout abort),
        2. first 4 generated tokens of the 36-step run equal the 4 tokens of the
           4-step baseline run (`prefix4_match == True`),
        3. all token IDs in `[0, vocab)` and `per_pair_dt > 0`.
    """
    if requested_workers < 1 or requested_workers > 64:
        raise ValueError(f"R3 violation: workers={requested_workers} must be in [1, 64]")

    sha = shard_sha256(ckpt_dir)
    t0_pack = time.perf_counter()
    bank, off, _ = pack_mid_bank(ckpt_dir)
    pack_s = time.perf_counter() - t0_pack
    bank_gib = bank.nbytes / (2 ** 30)

    lib = load_lib()
    occ = query_occupancy(lib, mode=mode)
    cotenants = _probe_cotenants()

    max_total = 600
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(max_total))

    rng = np.random.default_rng(0)
    prompt512 = rng.integers(3, MID_VOCAB, size=512, dtype=np.int32)
    prompt512_sha256 = hashlib.sha256(prompt512.tobytes()).hexdigest()
    prompt4 = np.array([1, 5, 6, 7], dtype=np.int32)
    prompt4_sha256 = hashlib.sha256(prompt4.tobytes()).hexdigest()

    build_names = {
        0: "bf16-production (mid_generate, kv_dtype=bf16)",
        1: "fp32-compute-bf16kv (mid_generate_fp32_bf16kv, kv_dtype=bf16)",
        2: "fp32-validation (mid_generate_fp32kv, kv_dtype=fp32)",
    }

    def run_one(p_arr: np.ndarray, n_out: int, w: int) -> tuple[float, list[int], list[int], int]:
        out_buf = np.zeros(n_out, dtype=np.int32)
        ctrl = np.zeros(4, dtype=np.uint32)
        t0 = time.perf_counter()
        rc = lib.mid_session_run_ex(
            sess,
            p_arr.ctypes.data,
            ctypes.c_int(len(p_arr)),
            out_buf.ctypes.data,
            ctypes.c_int(n_out),
            None,
            None,
            ctypes.c_int(w),
            ctypes.c_int(mode),
            ctrl.ctypes.data,
        )
        dt = time.perf_counter() - t0
        return dt, out_buf.tolist(), [int(x) for x in ctrl], int(rc)

    def bench_at_workers(w: int) -> dict:
        # Probe / warmup: 2 short launches (prompt4, n_out=8)
        for _ in range(2):
            dt, toks, ctrl, rc = run_one(prompt4, 8, w)
            if rc != 0 or ctrl[2] != 0 or ctrl[1] == 0 or not any(t != 0 for t in toks):
                return {
                    "workers": w,
                    "valid": False,
                    "abort_reason": f"warmup launch/barrier failure: rc={rc}, control={ctrl} (phase={ctrl[1]}, abort_flag={ctrl[2]})",
                    "warmup_elapsed_s": round(dt, 4),
                }

        # 1. Short-ctx ABBA over 4 pairs: A=(prompt4, n_out=4), B=(prompt4, n_out=36)
        short_pairs = []
        abba_orders = ["AB", "BA", "AB", "BA"]
        for p_idx, order in enumerate(abba_orders):
            if order == "AB":
                t_a, tok_a, ctrl_a, rc_a = run_one(prompt4, 4, w)
                t_b, tok_b, ctrl_b, rc_b = run_one(prompt4, 36, w)
            else:
                t_b, tok_b, ctrl_b, rc_b = run_one(prompt4, 36, w)
                t_a, tok_a, ctrl_a, rc_a = run_one(prompt4, 4, w)
            dt_pair = t_b - t_a
            prefix_ok = (tok_a == tok_b[:4])
            ids_ok = all(0 <= t < MID_VOCAB for t in tok_b) and any(t != 0 for t in tok_b)
            no_abort = (
                rc_a == 0 and rc_b == 0
                and ctrl_a[2] == 0 and ctrl_b[2] == 0
                and ctrl_a[1] > 0 and ctrl_b[1] > 0
            )
            pair_valid = no_abort and prefix_ok and ids_ok and (dt_pair > 0)
            short_pairs.append({
                "pair": p_idx,
                "order": order,
                "t8_s": round(t_a, 5),
                "t40_s": round(t_b, 5),
                "dt32_s": round(dt_pair, 5),
                "decode32_tok_s": round(32.0 / dt_pair, 2) if dt_pair > 0 else 0.0,
                "control_a_abort": ctrl_a[2],
                "control_b_abort": ctrl_b[2],
                "prefix4_match": prefix_ok,
                "ids_in_vocab": ids_ok,
                "valid": pair_valid,
            })

        # 2. 512-ctx ABBA over 4 pairs after 1 warmup at 516
        w_dt, w_tok, w_ctrl, w_rc = run_one(prompt512, 4, w)
        if w_rc != 0 or w_ctrl[2] != 0 or w_ctrl[1] == 0 or not any(t != 0 for t in w_tok):
            return {
                "workers": w,
                "valid": False,
                "abort_reason": f"512-ctx warmup launch/barrier failure: rc={w_rc}, control={w_ctrl}",
            }

        ctx512_pairs = []
        last_tok36 = []
        for p_idx, order in enumerate(abba_orders):
            if order == "AB":
                t_a, tok_a, ctrl_a, rc_a = run_one(prompt512, 4, w)
                t_b, tok_b, ctrl_b, rc_b = run_one(prompt512, 36, w)
            else:
                t_b, tok_b, ctrl_b, rc_b = run_one(prompt512, 36, w)
                t_a, tok_a, ctrl_a, rc_a = run_one(prompt512, 4, w)
            last_tok36 = tok_b
            dt_pair = t_b - t_a
            prefix_ok = (tok_a == tok_b[:4])
            ids_ok = all(0 <= t < MID_VOCAB for t in tok_b) and any(t != 0 for t in tok_b)
            no_abort = (
                rc_a == 0 and rc_b == 0
                and ctrl_a[2] == 0 and ctrl_b[2] == 0
                and ctrl_a[1] > 0 and ctrl_b[1] > 0
            )
            pair_valid = no_abort and prefix_ok and ids_ok and (dt_pair > 0)
            ctx512_pairs.append({
                "pair": p_idx,
                "order": order,
                "t516_s": round(t_a, 5),
                "t548_s": round(t_b, 5),
                "dt32_s": round(dt_pair, 5),
                "decode32_tok_s": round(32.0 / dt_pair, 2) if dt_pair > 0 else 0.0,
                "control_a_abort": ctrl_a[2],
                "control_b_abort": ctrl_b[2],
                "prefix4_match": prefix_ok,
                "ids_in_vocab": ids_ok,
                "valid": pair_valid,
            })

        all_valid = all(p["valid"] for p in short_pairs) and all(p["valid"] for p in ctx512_pairs)
        t8_mean = sum(p["t8_s"] for p in short_pairs) / 4.0
        t40_mean = sum(p["t40_s"] for p in short_pairs) / 4.0
        dt_short_mean = t40_mean - t8_mean
        short_rates = [p["decode32_tok_s"] for p in short_pairs]

        t516_mean = sum(p["t516_s"] for p in ctx512_pairs) / 4.0
        t548_mean = sum(p["t548_s"] for p in ctx512_pairs) / 4.0
        dt_512_mean = t548_mean - t516_mean
        ctx512_rates = [p["decode32_tok_s"] for p in ctx512_pairs]

        return {
            "workers": w,
            "valid": all_valid,
            "decode32_after_512_differential_tok_s": round(32.0 / dt_512_mean, 2) if dt_512_mean > 0 else 0.0,
            "decode32_after_512_pair_min_tok_s": round(min(ctx512_rates), 2),
            "decode32_after_512_pair_max_tok_s": round(max(ctx512_rates), 2),
            "decode32_short_ctx_differential_tok_s": round(32.0 / dt_short_mean, 2) if dt_short_mean > 0 else 0.0,
            "decode32_short_ctx_pair_min_tok_s": round(min(short_rates), 2),
            "decode32_short_ctx_pair_max_tok_s": round(max(short_rates), 2),
            "end_to_end_516_pos_tok_s": round(516.0 / t516_mean, 2),
            "t516_mean_s": round(t516_mean, 4),
            "t548_mean_s": round(t548_mean, 4),
            "dt512_mean_s": round(dt_512_mean, 4),
            "t8_mean_s": round(t8_mean, 4),
            "t40_mean_s": round(t40_mean, 4),
            "dt_short_mean_s": round(dt_short_mean, 4),
            "ctx512_pairs": ctx512_pairs,
            "short_pairs": short_pairs,
            "generated_tokens_sample": last_tok36[:12],
        }

    runs = []
    # 1. Run at requested_workers (e.g. 64)
    r_req = bench_at_workers(requested_workers)
    runs.append(r_req)

    # 2. If requested_workers > max_co_resident_workers or > cus, also run at clamped occupancy & CU count
    clamped_w = min(requested_workers, occ["max_co_resident_workers"])
    if clamped_w != requested_workers:
        runs.append(bench_at_workers(clamped_w))
    if occ["cus"] not in (requested_workers, clamped_w):
        runs.append(bench_at_workers(occ["cus"]))

    lib.mid_session_free(sess)

    # Session peak memory: bank + sliding KV (bf16) + full KV@600 (bf16) + ws_f + logits
    session_peak_gib = bank_gib + (
        4 * 2 * MID_NKV * MID_WIN * MID_HDIM * 2
        + 4 * 2 * MID_NKV * max_total * MID_HDIM * 2
        + (32768 + 128 + MID_VOCAB) * 4
        + MID_VOCAB * 4
    ) / (2 ** 30)

    res = {
        "protocol": "rec-definitions-R1-R8 differential ABBA (4 pairs)",
        "ckpt": str(ckpt_dir),
        "ckpt_sha256": sha,
        "build_tier": build_names.get(mode, f"mode={mode}"),
        "prompt512_expr": "np.random.default_rng(0).integers(3, 150144, size=512, dtype=np.int32)",
        "prompt512_int32_sha256": prompt512_sha256,
        "prompt4": [1, 5, 6, 7],
        "prompt4_int32_sha256": prompt4_sha256,
        "occupancy": occ,
        "co_tenants": cotenants,
        "resident_bank_gib": round(bank_gib, 3),
        "session_peak_gib": round(session_peak_gib, 3),
        "pack_seconds": round(pack_s, 3),
        "runs": runs,
    }
    print(json.dumps(res, indent=2))


def run_launch_failure_test(ckpt_dir: Path) -> bool:
    """Regression test for rec-tan.5:
      1. Unit-test `assert_valid_launch_outputs` against unwritten/no-op outputs.
      2. Subprocess test with `HSA_OVERRIDE_GFX_VERSION=11.0.0` (which hides the gfx1150
         code object in `flint_mid.so`): verify `mid_session_run_ex2` returns a negative
         rc (`-6` from `hipLaunchKernel` / `hipGetLastError()` or `-7` from `control[1]==0`)
         and `mid_mk_driver.py` exits nonzero instead of writing all-zero outputs.
      3. Subprocess test with `HSA_OVERRIDE_GFX_VERSION` unset: verify `mid_session_run_ex2`
         returns `rc == 0`, `control[1] > 0`, `control[2] == 0`, and non-zero valid outputs.
    """
    import os
    import subprocess

    # 1. Unit assertions on assert_valid_launch_outputs
    z_tok = np.zeros(4, dtype=np.int32)
    nz_tok = np.array([124538, 146358, 17939, 145985], dtype=np.int32)
    z_ctrl = np.array([0, 0, 0, 0], dtype=np.uint32)
    ok_ctrl = np.array([0, 320, 0, 0], dtype=np.uint32)

    for label, args in [
        ("nonzero_rc", (-6, ok_ctrl, nz_tok)),
        ("zero_phase_no_op", (0, z_ctrl, nz_tok)),
        ("all_zero_tokens", (0, ok_ctrl, z_tok)),
    ]:
        caught = False
        try:
            assert_valid_launch_outputs(*args)
        except RuntimeError:
            caught = True
        assert caught, f"rec-tan.5 unit check failed to reject {label}"
    print("rec-tan.5 unit check: assert_valid_launch_outputs rejected all invalid cases -> PASS")

    # 2. Subprocess check with HSA_OVERRIDE_GFX_VERSION=11.0.0 (must fail with rc < 0)
    env_bad = dict(os.environ)
    env_bad["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    probe_code = (
        "import ctypes, sys, numpy as np\n"
        "from pathlib import Path\n"
        "import mid_mk_driver as D\n"
        f"ckpt = Path({str(ckpt_dir)!r})\n"
        "bank, off, _ = D.pack_mid_bank(ckpt)\n"
        "lib = D.load_lib()\n"
        "sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(64))\n"
        "p = np.array([1, 43, 1006, 52006], dtype=np.int32)\n"
        "out = np.zeros(4, dtype=np.int32)\n"
        "logits = np.zeros(D.MID_VOCAB, dtype=np.float32)\n"
        "ctrl = np.zeros(4, dtype=np.uint32)\n"
        "rc = lib.mid_session_run_ex2(sess, p.ctypes.data, 4, out.ctypes.data, 4, None, logits.ctypes.data, 16, 2, ctrl.ctypes.data, None)\n"
        "lib.mid_session_free(sess)\n"
        "print(f'PROBE rc={rc} ctrl={ctrl.tolist()} tokens={out.tolist()}')\n"
        "sys.exit(0 if rc == 0 else 42)\n"
    )
    proc_bad = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(HERE),
        env=env_bad,
        capture_output=True,
        text=True,
    )
    print("rec-tan.5 HSA_OVERRIDE_GFX_VERSION=11.0.0 stdout:", proc_bad.stdout.strip())
    print("rec-tan.5 HSA_OVERRIDE_GFX_VERSION=11.0.0 stderr:", proc_bad.stderr.strip())
    assert proc_bad.returncode == 42, (
        f"rec-tan.5 regression: expected rc != 0 (exit 42) under HSA_OVERRIDE_GFX_VERSION=11.0.0, got exit {proc_bad.returncode}"
    )
    print("rec-tan.5 bad-override launch check: rejected with nonzero rc -> PASS")
    return True


def run_q4_bench() -> bool:
    """Validate and benchmark standalone fused INT4 dequant-matmul (`flint_q4_gemv_host`)
    on Flint production shapes:
      - Q projection: [4096, 2048]
      - O projection: [2048, 4096]
      - Routed expert gate/up: [512, 2048]
      - LM Head: [150144, 2048]
    """
    lib = ctypes.CDLL(str(HERE / "flint_mid.so"))
    lib.flint_q4_gemv_host.restype = ctypes.c_int
    lib.flint_q4_gemv_host.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    rng = np.random.default_rng(42)
    shapes = [
        ("expert_gate_512x2048", 512, 2048),
        ("q_proj_4096x2048", 4096, 2048),
        ("o_proj_2048x4096", 2048, 4096),
        ("lm_head_150144x2048", 150144, 2048),
    ]
    all_ok = True
    for label, n, k in shapes:
        nib = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
        s_f = ((rng.random((n, k // 64)) * 0.5 + 0.75) * (0.13 / np.sqrt(k))).astype(np.float32)
        b_f = (-7.5 * s_f * (1.0 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32)
        x_f = rng.standard_normal(k).astype(np.float32)

        s_bf = f32_to_bf16(s_f)
        b_bf = f32_to_bf16(b_f)
        x_bf = f32_to_bf16(x_f)
        packed = np.ascontiguousarray((nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8))

        out_f = np.zeros(n, dtype=np.float32)
        ms = ctypes.c_float(0.0)
        rc = lib.flint_q4_gemv_host(
            packed.ctypes.data,
            s_bf.ctypes.data,
            b_bf.ctypes.data,
            x_bf.ctypes.data,
            None,
            out_f.ctypes.data,
            ctypes.c_int(n),
            ctypes.c_int(k),
            ctypes.c_int(0),
            ctypes.c_int(10),
            ctypes.c_int(50),
            ctypes.byref(ms),
        )
        assert rc == 0, f"flint_q4_gemv_host failed with {rc}"

        # Reference dequant + dot
        s_deq = bf16_to_f32(s_bf).astype(np.float64)
        b_deq = bf16_to_f32(b_bf).astype(np.float64)
        x_deq = bf16_to_f32(x_bf).astype(np.float64)
        w_deq = np.repeat(s_deq, 64, axis=1) * nib.astype(np.float64) + np.repeat(b_deq, 64, axis=1)
        ref = (w_deq @ x_deq).astype(np.float32)

        max_err = float(np.abs(out_f - ref).max())
        rel_err = max_err / max(float(np.abs(ref).max()), 1e-6)
        bytes_read = packed.nbytes + s_bf.nbytes + b_bf.nbytes + x_bf.nbytes + out_f.nbytes
        gb_s = (bytes_read / 1e9) / (ms.value / 1e3) if ms.value > 0 else 0.0
        ok = rel_err < 2e-3
        all_ok = all_ok and ok
        print(
            f"q4_gemv {label:22s}: {ms.value:7.3f} ms | {gb_s:6.2f} GB/s | "
            f"max|err| {max_err:.3e} rel {rel_err:.3e} -> {'PASS' if ok else 'FAIL'}"
        )
    return all_ok


def main(argv: list[str]) -> int:
    ckpt: Path | None = None
    prompt = [1, 5, 6, 7]
    n_out = 16
    workers = 64
    mode = 0
    out = HERE / "mid_mk_io.npz"
    verify_path: Path | None = None
    emit_suite_dir: Path | None = None
    suite_builds = ("fp32kv", "bf16kv", "fp32_bf16kv")
    suite_cases: tuple[str, ...] | None = None
    emit_experts = False
    bench = False
    q4_bench = False
    test_launch_check = False

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--bench":
            bench = True
            i += 1
            continue
        if a == "--q4-bench":
            q4_bench = True
            i += 1
            continue
        if a == "--test-launch-check":
            test_launch_check = True
            i += 1
            continue
        if a == "--emit-experts":
            emit_experts = True
            i += 1
            continue
        if "=" in a:
            key, _, val = a.partition("=")
        else:
            key = a
            val = argv[i + 1] if i + 1 < len(argv) else ""
            i += 1
        if key == "--ckpt":
            ckpt = Path(val)
        elif key == "--prompt":
            prompt = [int(t) for t in val.split(",")]
        elif key == "--n-out":
            n_out = int(val)
        elif key == "--workers":
            workers = int(val)
        elif key == "--mode":
            mode = int(val)
        elif key == "--out":
            out = Path(val)
        elif key == "--verify":
            verify_path = Path(val)
        elif key == "--emit-suite":
            emit_suite_dir = Path(val)
        elif key == "--builds":
            suite_builds = tuple(val.split(","))
        elif key == "--cases":
            suite_cases = tuple(val.split(","))
        elif key == "--emit-experts":
            emit_experts = val.lower() not in ("0", "false", "no", "off")
        else:
            raise SystemExit(f"unknown arg {a}")
        i += 1

    if q4_bench:
        return 0 if run_q4_bench() else 1

    if ckpt is None:
        raise SystemExit("--ckpt <checkpoint_dir> is required (no host-specific default)")

    if test_launch_check:
        return 0 if run_launch_failure_test(ckpt) else 1

    if emit_suite_dir is not None:
        emit_conformance_suite(
            ckpt, emit_suite_dir, workers=workers, builds=suite_builds, emit_experts=emit_experts,
            only_cases=suite_cases,
        )
        return 0

    if verify_path is not None:
        return 0 if verify_run(verify_path, ckpt) else 1

    if bench:
        run_benchmark(ckpt, requested_workers=workers, mode=mode)
        return 0

    bank, off, _ = pack_mid_bank(ckpt)
    prompt_arr = np.array(prompt, dtype=np.int32)
    p_len = len(prompt_arr)
    out_tokens = np.zeros(n_out, dtype=np.int32)
    out_logits = np.zeros(MID_VOCAB, dtype=np.float32)
    all_experts = (
        np.full((p_len + n_out, MID_NL, MID_TOPK), -1, dtype=np.int32)
        if emit_experts
        else None
    )
    ctrl = np.zeros(4, dtype=np.uint32)

    lib = load_lib()
    max_total = max(p_len + n_out + 16, 64)
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(max_total))
    t0 = time.perf_counter()
    rc = lib.mid_session_run_ex2(
        sess,
        prompt_arr.ctypes.data,
        ctypes.c_int(p_len),
        out_tokens.ctypes.data,
        ctypes.c_int(n_out),
        None,
        out_logits.ctypes.data,
        ctypes.c_int(workers),
        ctypes.c_int(mode),
        ctrl.ctypes.data,
        all_experts.ctypes.data if all_experts is not None else None,
    )
    elapsed = time.perf_counter() - t0
    lib.mid_session_free(sess)
    assert_valid_launch_outputs(rc, ctrl, out_tokens, out_logits, None, all_experts)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(
        prompt=prompt_arr,
        tokens=out_tokens,
        last_logits=out_logits,
        workers=np.int32(workers),
        elapsed_s=np.float64(elapsed),
    )
    if all_experts is not None:
        save_kw["chosen_experts"] = all_experts[p_len - 1 : p_len - 1 + n_out]
        save_kw["step_experts"] = all_experts
    np.savez(out, **save_kw)
    print(f"tokens ({n_out} in {elapsed*1e3:.1f} ms, {(len(prompt)+n_out)/elapsed:.1f} step/s):", out_tokens.tolist())
    print(
        f"logits: min {out_logits.min():.4f} max {out_logits.max():.4f} "
        f"argmax {int(out_logits.argmax())} absmean {np.abs(out_logits).mean():.4f} "
        f"finite {bool(np.isfinite(out_logits).all())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
