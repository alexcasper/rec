"""Pack a tiny-Flint weight bank (numpy, seeded), launch the persistent
megakernel in a torch-free process, save inputs + token outputs to npz.

    python tiny_mk_driver.py [--seed=0] [--data=signed|legacy]
                             [--prompt=1,5,6,7] [--n-out=8] [--out=tiny_mk_io.npz]

Then check with tiny_mk_np_reference.py (any host) and/or
tiny_mk_mlx_reference.py (production model code; ground truth).
make_bank() is importable without a GPU (tests/test_tiny_mk_reference.py).
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HID, HEADS, NKV, HDIM = 128, 4, 2, 64
NEXP, MOE_I, SH_I, DENSE_I, VOCAB = 8, 64, 64, 256, 512

# bank tensor order -- mirrors tiny_megakernel.hip IDX_* / Q4 tables
# bf16 order: embed, L0(ln,post,qn,kn,sinks), L1(...), router w/b, final norm
ORDER_BF16 = [("embed", VOCAB * HID), ("l0_ln", HID), ("l0_post", HID),
              ("l0_qn", HDIM), ("l0_kn", HDIM), ("l0_sinks", HEADS),
              ("l1_ln", HID), ("l1_post", HID), ("l1_qn", HDIM), ("l1_kn", HDIM),
              ("l1_sinks", HEADS), ("router_w", NEXP * HID), ("router_b", NEXP),
              ("fn_norm", HID)]
Q4_ORDER = [("l0_q", HEADS * HDIM, HID), ("l0_k", NKV * HDIM, HID),
            ("l0_v", NKV * HDIM, HID), ("l0_o", HID, HEADS * HDIM),
            ("l0_gate", DENSE_I, HID), ("l0_up", DENSE_I, HID), ("l0_down", HID, DENSE_I),
            ("l1_q", HEADS * HDIM, HID), ("l1_k", NKV * HDIM, HID),
            ("l1_v", NKV * HDIM, HID), ("l1_o", HID, HEADS * HDIM),
            ("l1_sw_gate", NEXP * MOE_I, HID), ("l1_sw_up", NEXP * MOE_I, HID),
            ("l1_sw_down", NEXP * HID, MOE_I),
            ("l1_sh_gate", SH_I, HID), ("l1_sh_up", SH_I, HID), ("l1_sh_down", HID, SH_I),
            ("lm_head", VOCAB, HID)]


def f32_to_bf16(v):
    u = v.astype(np.float32).view(np.uint32)
    return ((u + np.uint32(0x8000)) >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def make_bank(seed=0, data="signed"):
    """Return (bank uint8[], offsets int64[], {"w_<name>": array}).

    data="signed" (default) mimics a real affine-quantized checkpoint:
    zero-centred weights (bias ~ -7.5*scale), std ~0.6/sqrt(K), so activations
    stay O(1), logits are spread and greedy tokens vary -- a test that can
    tell SiLU from identity or a mis-wired residual. data="legacy" is the
    original all-nonnegative bank (scale 0.5-1.5, bias +-0.1): every matmul is
    a large positive sum, logits ~1e3, one token repeats forever, and semantic
    bugs hide (rec-hjr follow-up). Kept only to reproduce old runs.
    """
    assert data in ("signed", "legacy"), data
    rng = np.random.default_rng(seed)

    def quantize(n, k):
        nib = rng.integers(0, 16, size=(n, k)).astype(np.uint8)
        if data == "legacy":
            s = (rng.random((n, k // 64)) * 1.0 + 0.5).astype(np.float32)
            b = (rng.random((n, k // 64)) * 0.2 - 0.1).astype(np.float32)
        else:
            s = ((rng.random((n, k // 64)) * 0.5 + 0.75) * (0.13 / np.sqrt(k))).astype(np.float32)
            b = (-7.5 * s * (1.0 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32)
        # round s/b to bf16 BEFORE dequantizing so deq is exactly what the kernel reads
        s, b = bf16_to_f32(f32_to_bf16(s)), bf16_to_f32(f32_to_bf16(b))
        deq = np.repeat(s, 64, axis=1) * nib + np.repeat(b, 64, axis=1)
        packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
        return packed.reshape(-1), f32_to_bf16(s.reshape(-1)), f32_to_bf16(b.reshape(-1)), deq

    bank, offsets, store = bytearray(), [], {}
    for name, elems in ORDER_BF16:
        offsets.append(len(bank))
        v = (rng.standard_normal(elems).astype(np.float32) if name in ("embed", "router_w")
             else rng.random(elems).astype(np.float32) + 0.5)
        if name == "router_b":
            v = rng.random(elems).astype(np.float32) * 0.2 - 0.1
        if data == "signed" and name == "router_w":
            v = v * 0.1
        bits = f32_to_bf16(v)
        bank.extend(bits.tobytes())
        store["w_" + name] = bits  # bf16 bits
    for name, n, k in Q4_ORDER:
        offsets.append(len(bank))
        p, s, b, deq = quantize(n, k)
        for part in (p, s, b):
            bank.extend(part.tobytes())
        store["w_" + name] = deq  # fp32 [n, K]
    return np.frombuffer(bytes(bank), dtype=np.uint8).copy(), np.array(offsets, dtype=np.int64), store


def main(argv):
    data, seed, prompt, n_out, out = "signed", 0, [1, 5, 6, 7], 8, HERE / "tiny_mk_io.npz"
    for a in argv:
        key, _, val = a.partition("=")
        if key == "--data":
            data = val
        elif key == "--seed":
            seed = int(val)
        elif key == "--prompt":
            prompt = [int(t) for t in val.split(",")]
        elif key == "--n-out":
            n_out = int(val)
        elif key == "--out":
            out = Path(val)
        else:
            raise SystemExit(f"unknown arg {a}")
    bank, off, store = make_bank(seed, data)

    total = len(prompt) + n_out
    kv = np.zeros(2 * NKV * 2 * total * HDIM, dtype=np.uint16)  # [NL][NKV][2][total][HDIM] bf16
    out_tokens = np.zeros(n_out, dtype=np.int32)
    out_logits = np.zeros(VOCAB, dtype=np.float32)
    prompt_arr = np.array(prompt, dtype=np.int32)

    lib = ctypes.CDLL(str(HERE / "flint_tiny.so"))  # before any torch import (libhsa)
    lib.tiny_launch.restype = None
    lib.tiny_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] + [ctypes.c_void_p] + [ctypes.c_int] + [ctypes.c_void_p]
    lib.tiny_launch(bank.ctypes.data, off.ctypes.data, kv.ctypes.data,
                    prompt_arr.ctypes.data, ctypes.c_int(len(prompt)),
                    out_tokens.ctypes.data, ctypes.c_int(n_out), out_logits.ctypes.data)
    np.savez(out, bank=bank, offsets=off, tokens=out_tokens, last_logits=out_logits,
             data=np.array(data), seed=np.int32(seed),
             prompt=prompt_arr, n_out=np.int32(n_out), **store)
    lg = out_logits
    print("tokens:", out_tokens.tolist())
    print(f"logits: min {lg.min():.4f} max {lg.max():.4f} argmax {int(lg.argmax())} "
          f"absmean {np.abs(lg).mean():.4f} nonzero {int((lg != 0).sum())}/{VOCAB}")


if __name__ == "__main__":
    main(sys.argv[1:])
