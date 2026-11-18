"""Pack a tiny-Flint weight bank (numpy, seed 0), launch the persistent
megakernel in a torch-free process, save inputs + token outputs to npz."""
import ctypes, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tiny_mk_io.npz"

def f32_to_bf16(v):
    u = v.astype(np.float32).view(np.uint32)
    return ((u + np.uint32(0x8000)) >> 16).astype(np.uint16)

rng = np.random.default_rng(0)
HID, HEADS, NKV, HDIM = 128, 4, 2, 64
NEXP, MOE_I, SH_I, DENSE_I, VOCAB = 8, 64, 64, 256, 512

# bank tensor table (name, kind, n, K) — mirrors tiny_megakernel.hip
tensors = [("embed", "bf16", VOCAB * HID)]
def add_layer(idx_l, rotary_sink):
    pass
bf16_names = []
# bf16 order: embed, L0(ln,post,qn,kn,sinks), L1(...), router w/b, final norm
order_bf16 = [("embed", VOCAB * HID), ("l0_ln", HID), ("l0_post", HID),
              ("l0_qn", HDIM), ("l0_kn", HDIM), ("l0_sinks", HEADS),
              ("l1_ln", HID), ("l1_post", HID), ("l1_qn", HDIM), ("l1_kn", HDIM),
              ("l1_sinks", HEADS), ("router_w", NEXP * HID), ("router_b", NEXP),
              ("fn_norm", HID)]
q4_order = [("l0_q", HEADS * HDIM, HID), ("l0_k", NKV * HDIM, HID),
            ("l0_v", NKV * HDIM, HID), ("l0_o", HID, HEADS * HDIM),
            ("l0_gate", DENSE_I, HID), ("l0_up", DENSE_I, HID), ("l0_down", HID, DENSE_I),
            ("l1_q", HEADS * HDIM, HID), ("l1_k", NKV * HDIM, HID),
            ("l1_v", NKV * HDIM, HID), ("l1_o", HID, HEADS * HDIM),
            ("l1_sw_gate", NEXP * MOE_I, HID), ("l1_sw_up", NEXP * MOE_I, HID),
            ("l1_sw_down", NEXP * HID, MOE_I),
            ("l1_sh_gate", SH_I, HID), ("l1_sh_up", SH_I, HID), ("l1_sh_down", HID, SH_I),
            ("lm_head", VOCAB, HID)]

def quantize(n, K):
    """random INT4-affine: returns packed uint8 [n*K/2], s bf16bits, b bf16bits, deq fp32"""
    nib = rng.integers(0, 16, size=(n, K)).astype(np.uint8)
    s = (rng.random((n, K // 64)) * 1.0 + 0.5).astype(np.float32)
    b = (rng.random((n, K // 64)) * 0.2 - 0.1).astype(np.float32)
    sfull = np.repeat(s, 64, axis=1)
    bfull = np.repeat(b, 64, axis=1)
    deq = sfull * nib + bfull
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return packed.reshape(-1), f32_to_bf16(s.reshape(-1)), f32_to_bf16(b.reshape(-1)), deq

bank = bytearray()
offsets = []
deq_store = {}
for name, elems in order_bf16:
    offsets.append(len(bank))
    v = (rng.standard_normal(elems).astype(np.float32) if "embed" in name or "router_w" == name
         else rng.random(elems).astype(np.float32) + 0.5)
    if name == "router_b":
        v = (rng.random(elems).astype(np.float32) * 0.2 - 0.1)
    vbits = f32_to_bf16(v)
    bank.extend(vbits.tobytes())
    deq_store[name] = vbits  # bf16 bits
for name, n, K in q4_order:
    offsets.append(len(bank))
    p, s, b, deq = quantize(n, K)
    bank.extend(p.tobytes()); bank.extend(s.tobytes()); bank.extend(b.tobytes())
    deq_store[name] = deq  # fp32 [n, K]

bank_arr = np.frombuffer(bytes(bank), dtype=np.uint8).copy()
off_arr = np.array(offsets, dtype=np.int64)

prompt = [1, 5, 6, 7]
n_out = 8

# kv cache: [NL*NKV*2*total*HDIM] bf16
total = len(prompt) + n_out
kv = np.zeros(2 * NKV * 2 * total * HDIM, dtype=np.uint16)
out_tokens = np.zeros(n_out, dtype=np.int32)
out_logits = np.zeros(VOCAB, dtype=np.float32)

lib = ctypes.CDLL(str(HERE / "flint_tiny.so"))
lib.tiny_launch.restype = None
lib.tiny_launch.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int] + [ctypes.c_void_p] + [ctypes.c_int] + [ctypes.c_void_p]
prompt_arr = np.array(prompt, dtype=np.int32)
lib.tiny_launch(bank_arr.ctypes.data, off_arr.ctypes.data, kv.ctypes.data,
                       np.array(prompt, dtype=np.int32).ctypes.data, ctypes.c_int(len(prompt)),
                       out_tokens.ctypes.data, ctypes.c_int(n_out),
                       out_logits.ctypes.data)
np.savez(OUT, bank=bank_arr, offsets=off_arr, tokens=out_tokens,
         prompt=np.array(prompt, dtype=np.int32), n_out=np.int32(n_out),
         **{("w_" + k): v for k, v in deq_store.items()})
print("tokens:", out_tokens.tolist())
lg = out_logits
mn, mx, am = float(lg.min()), float(lg.max()), int(lg.argmax())
abm = float(np.abs(lg).mean()); nz = int((lg != 0).sum())
print(f"logits: min {mn:.4f} max {mx:.4f} argmax {am} absmean {abm:.4f} nonzero {nz}/512")
