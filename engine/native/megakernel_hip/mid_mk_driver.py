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


def forward_mid_np(raw: dict[str, tuple[np.ndarray, str]], seq: list[int]) -> np.ndarray:
    """Portable fp64 numpy reference forward for mid-8L over `seq` (returns [T, V] fp32 logits).
    Mirrors tiny_mk_np_reference.py / rec_mlx.model.Model semantics exactly."""
    T = len(seq)
    seq_arr = np.array(seq, dtype=np.int64)
    x = dequant_np(raw, "model.embed_tokens", bits=8, idx=seq_arr)

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
    return (fn @ w_lm.T).astype(np.float32)


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
    lib.mid_session_create.restype = ctypes.c_void_p
    lib.mid_session_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mid_session_run.restype = None
    lib.mid_session_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.mid_session_free.restype = None
    lib.mid_session_free.argtypes = [ctypes.c_void_p]
    return lib


def run_benchmark(ckpt_dir: Path, workers: int = 16) -> None:
    """Measure mid-8L persistent megakernel decode32 tok/s and 512-token sequence throughput.

    Since `mid_generate` is a persistent single-launch token-loop kernel, a run of
    (prompt_len=P, n_out=N) executes P+N steps end-to-end in one kernel launch.
    To isolate pure decode32 tok/s after a 512-token context (matching run-definitions.md:
    512 prefill + 4 warmup decode + 32 timed decode steps), we measure:
      - `t_516`: wall-clock for 512 prompt + 4 warmup decode steps (516 total positions)
      - `t_548`: wall-clock for 512 prompt + 36 decode steps (548 total positions)
      - `decode32_after_512_tok_s = 32 / (t_548 - t_516)`
    And we also measure direct short-prompt decode32 (`prompt=4, n_out=32`) and
    single-launch 512-token throughput (`512 / t_512`).
    """
    t0_pack = time.perf_counter()
    bank, off, _ = pack_mid_bank(ckpt_dir)
    pack_s = time.perf_counter() - t0_pack
    bank_gib = bank.nbytes / (2 ** 30)

    lib = load_lib()
    max_total = 600
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(max_total))

    rng = np.random.default_rng(0)
    prompt512 = rng.integers(3, MID_VOCAB, size=512, dtype=np.int32)
    prompt4 = np.array([1, 5, 6, 7], dtype=np.int32)
    out_buf = np.zeros(64, dtype=np.int32)

    # Warmup launches
    for _ in range(2):
        lib.mid_session_run(sess, prompt4.ctypes.data, 4, out_buf.ctypes.data, 8, None, workers)

    # 1. Short-prompt decode32 (prompt=4, n_out=32 -> 36 steps, vs prompt=4, n_out=4 -> 8 steps)
    t_short4_list, t_short36_list = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        lib.mid_session_run(sess, prompt4.ctypes.data, 4, out_buf.ctypes.data, 4, None, workers)
        t_short4_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lib.mid_session_run(sess, prompt4.ctypes.data, 4, out_buf.ctypes.data, 36, None, workers)
        t_short36_list.append(time.perf_counter() - t0)

    dt_short32 = (sum(t_short36_list) / 3.0) - (sum(t_short4_list) / 3.0)
    decode32_short_tok_s = 32.0 / dt_short32
    decode_all36_tok_s = 36.0 / (sum(t_short36_list) / 3.0)

    # 2. Full 512-context protocol (512 prompt + 4 warmup vs 512 prompt + 36 decode)
    # 1 warmup at 516
    lib.mid_session_run(sess, prompt512.ctypes.data, 512, out_buf.ctypes.data, 4, None, workers)
    t_516_list, t_548_list = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        lib.mid_session_run(sess, prompt512.ctypes.data, 512, out_buf.ctypes.data, 4, None, workers)
        t_516_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lib.mid_session_run(sess, prompt512.ctypes.data, 512, out_buf.ctypes.data, 36, None, workers)
        t_548_list.append(time.perf_counter() - t0)

    t516_mean = sum(t_516_list) / len(t_516_list)
    t548_mean = sum(t_548_list) / len(t_548_list)
    dt_ctx512_32 = t548_mean - t516_mean
    decode32_ctx512_tok_s = 32.0 / dt_ctx512_32 if dt_ctx512_32 > 0 else 0.0
    seq516_tok_s = 516.0 / t516_mean

    lib.mid_session_free(sess)

    res = {
        "ckpt": str(ckpt_dir),
        "kernel": "flint_mid_megakernel.hip (mid_generate, single-launch persistent INT4/INT8+BF16)",
        "workers": workers,
        "threads_per_worker": 256,
        "resident_bank_gib": round(bank_gib, 3),
        "pack_seconds": round(pack_s, 3),
        "decode32_short_ctx_tok_s": round(decode32_short_tok_s, 2),
        "decode36_end_to_end_tok_s": round(decode_all36_tok_s, 2),
        "decode32_after_512_prefill_tok_s": round(decode32_ctx512_tok_s, 2),
        "serial_516_step_mean_tok_s": round(seq516_tok_s, 2),
        "t516_s": round(t516_mean, 4),
        "t548_s": round(t548_mean, 4),
        "generated_tokens_sample": out_buf[:12].tolist(),
    }
    print(json.dumps(res, indent=2))


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
    ckpt = Path("/home/dusk/rec-ve/ckpt/cpu-baseline-mid")
    prompt = [1, 5, 6, 7]
    n_out = 16
    workers = 16
    out = HERE / "mid_mk_io.npz"
    verify_path: Path | None = None
    bench = False
    q4_bench = False

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
        elif key == "--out":
            out = Path(val)
        elif key == "--verify":
            verify_path = Path(val)
        else:
            raise SystemExit(f"unknown arg {a}")
        i += 1

    if q4_bench:
        return 0 if run_q4_bench() else 1

    if verify_path is not None:
        return 0 if verify_run(verify_path, ckpt) else 1

    if bench:
        run_benchmark(ckpt, workers=workers)
        return 0

    bank, off, _ = pack_mid_bank(ckpt)
    prompt_arr = np.array(prompt, dtype=np.int32)
    out_tokens = np.zeros(n_out, dtype=np.int32)
    out_logits = np.zeros(MID_VOCAB, dtype=np.float32)

    lib = load_lib()
    max_total = max(len(prompt) + n_out + 16, 64)
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(max_total))
    t0 = time.perf_counter()
    lib.mid_session_run(
        sess,
        prompt_arr.ctypes.data,
        ctypes.c_int(len(prompt)),
        out_tokens.ctypes.data,
        ctypes.c_int(n_out),
        out_logits.ctypes.data,
        ctypes.c_int(workers),
    )
    elapsed = time.perf_counter() - t0
    lib.mid_session_free(sess)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        prompt=prompt_arr,
        tokens=out_tokens,
        last_logits=out_logits,
        workers=np.int32(workers),
        elapsed_s=np.float64(elapsed),
    )
    print(f"tokens ({n_out} in {elapsed*1e3:.1f} ms, {(len(prompt)+n_out)/elapsed:.1f} step/s):", out_tokens.tolist())
    print(
        f"logits: min {out_logits.min():.4f} max {out_logits.max():.4f} "
        f"argmax {int(out_logits.argmax())} absmean {np.abs(out_logits).mean():.4f} "
        f"finite {bool(np.isfinite(out_logits).all())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
